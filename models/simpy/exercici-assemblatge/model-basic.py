#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Discrete-event simulation of a product assembly line using SimPy.

Product = Superior Cover + Inferior Cover + Interior Element

Flows
-----
1. Covers arrive Exp(11 min), 50/50 superior/inferior.
   Painted Uniform(6,12) min, 95% pass QC, 5% rework (re-enter FIFO queue).
2. Interior elements arrive in boxes of 3, Exp(64 min).
   Unpacked Uniform(30,50) min. 10% defective → scrap.
3. Assembly: 1 sup. + 1 inf. + 1 interior → finished product, Uniform(10,20) min.
"""

import simpy
import random
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Entity classes
# ---------------------------------------------------------------------------

@dataclass
class Cover:
    """A cover piece (superior or inferior)."""
    id: int
    cover_type: str  # 'superior' or 'inferior'
    created_at: float = 0.0
    paint_attempts: int = 0
    ready_at: Optional[float] = None

    def __str__(self) -> str:
        return f"Cover({self.id}, {self.cover_type})"


@dataclass
class InteriorElement:
    """An interior element unpacked from a box."""
    id: int
    box_id: int
    created_at: float = 0.0
    ready_at: Optional[float] = None

    def __str__(self) -> str:
        return f"InteriorElement({self.id}, box={self.box_id})"


@dataclass
class InteriorElementsBox:
    """A box containing 3 interior elements."""
    id: int
    created_at: float = 0.0
    num_elements: int = 3

    def __str__(self) -> str:
        return f"InteriorElementsBox({self.id})"


@dataclass
class FinalProduct:
    """A finished assembled product."""
    id: int
    superior_cover: Cover
    inferior_cover: Cover
    interior_element: InteriorElement
    assembled_at: float = 0.0

    def __str__(self) -> str:
        return (f"FinalProduct({self.id}: "
                f"{self.superior_cover}, {self.inferior_cover}, "
                f"{self.interior_element})")


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

class StationStats:
    """
    Tracks resource utilization, queue waiting times, and time-weighted
    average queue length for a single-server station.
    """

    def __init__(self):
        # Resource utilization
        self.total_busy_time: float = 0.0
        self._busy_start: Optional[float] = None

        # Queue waiting time
        self.total_wait_time: float = 0.0
        self.wait_count: int = 0
        self._wait_starts: dict = {}

        # Time-weighted queue length
        self.current_queue_length: int = 0
        self._last_change_time: float = 0.0
        self._total_area: float = 0.0

    def record_queue_entry(self, now: float, entity_id: int):
        """Called when an entity joins the queue (before yield req)."""
        self._update_area(now)
        self.current_queue_length += 1
        self._wait_starts[entity_id] = now

    def record_service_start(self, now: float, entity_id: int):
        """Called when service begins (after yield req)."""
        self._update_area(now)
        self.current_queue_length -= 1
        if entity_id in self._wait_starts:
            wait_time = now - self._wait_starts.pop(entity_id)
            self.total_wait_time += wait_time
            self.wait_count += 1
        self._busy_start = now

    def record_service_end(self, now: float):
        """Called when service completes."""
        if self._busy_start is not None:
            self.total_busy_time += now - self._busy_start
            self._busy_start = None

    def _update_area(self, now: float):
        self._total_area += self.current_queue_length * (now - self._last_change_time)
        self._last_change_time = now

    def utilization(self, sim_time: float) -> float:
        """Returns resource utilization as a fraction [0, 1]."""
        if sim_time <= 0:
            return 0.0
        return self.total_busy_time / sim_time

    def average_wait_time(self) -> float:
        """Returns average wait time in queue (minutes)."""
        if self.wait_count == 0:
            return 0.0
        return self.total_wait_time / self.wait_count

    def average_queue_length(self, sim_time: float) -> float:
        """Returns time-weighted average queue length."""
        if sim_time <= 0:
            return 0.0
        total = self._total_area + self.current_queue_length * (sim_time - self._last_change_time)
        return total / sim_time


class BufferStats:
    """
    Tracks time-weighted average length and maximum length for a simpy.Store buffer.
    """

    def __init__(self):
        self.current_length: int = 0
        self._last_change_time: float = 0.0
        self._total_area: float = 0.0
        self.max_length: int = 0

    def record_change(self, now: float, new_length: int):
        """Update area accumulator and current length."""
        self._total_area += self.current_length * (now - self._last_change_time)
        self._last_change_time = now
        self.current_length = new_length
        if new_length > self.max_length:
            self.max_length = new_length

    def average_length(self, sim_time: float) -> float:
        """Returns time-weighted average length."""
        if sim_time <= 0:
            return 0.0
        total = self._total_area + self.current_length * (sim_time - self._last_change_time)
        return total / sim_time


# ---------------------------------------------------------------------------
# Resource / Station classes
# ---------------------------------------------------------------------------

class PaintStation:
    """
    Painting station with a single server (SimPy Resource).
    Paint time ~ Uniform(6, 12) min.
    After painting, 95% pass QC; 5% return to the FIFO queue for rework.
    """

    def __init__(self, environ: simpy.Environment, capacity: int = 1):
        self.environ = environ
        self.resource = simpy.Resource(environ, capacity=capacity)
        self.painted_covers: int = 0
        self.reworked_covers: int = 0
        self.stats = StationStats()

    def paint(self, cover: Cover):
        """SimPy process: request resource, paint, check QC."""
        while True:
            cover.paint_attempts += 1
            with self.resource.request() as req:
                self.stats.record_queue_entry(self.environ.now, id(cover))
                yield req
                self.stats.record_service_start(self.environ.now, id(cover))
                paint_time = random.uniform(6, 12)
                yield self.environ.timeout(paint_time)
                self.stats.record_service_end(self.environ.now)
                self.painted_covers += 1

            # Quality control
            if random.random() < 0.95:
                # Passes QC
                cover.ready_at = self.environ.now
                return  # exit the loop – cover is good
            else:
                # Defective → rework (re-enters the FIFO queue)
                self.reworked_covers += 1
                print(f"  [{self.environ.now:7.2f}] {cover} REWORK "
                      f"(attempt {cover.paint_attempts})")


class UnpackingStation:
    """
    Unpacking station with a single machine (SimPy Resource).
    Unpack time ~ Uniform(30, 50) min per box of 3.
    10% of unpacked interior elements are defective → scrap.
    """

    def __init__(self, environ: simpy.Environment, capacity: int = 1):
        self.environ = environ
        self.resource = simpy.Resource(environ, capacity=capacity)
        self.boxes_unpacked: int = 0
        self.elements_ok: int = 0
        self.elements_scrap: int = 0
        self.stats = StationStats()

    def unpack(self, box: InteriorElementsBox):
        """SimPy process: unpack a box and return a list of good elements."""
        with self.resource.request() as req:
            self.stats.record_queue_entry(self.environ.now, id(box))
            yield req
            self.stats.record_service_start(self.environ.now, id(box))
            unpack_time = random.uniform(30, 50)
            yield self.environ.timeout(unpack_time)
            self.stats.record_service_end(self.environ.now)
            self.boxes_unpacked += 1

        # Inspect each element
        good_elements: list[InteriorElement] = []
        for i in range(box.num_elements):
            elem = InteriorElement(
                id=self.elements_ok + self.elements_scrap + 1,
                box_id=box.id,
                created_at=box.created_at,
            )
            if random.random() < 0.90:
                elem.ready_at = self.environ.now
                good_elements.append(elem)
                self.elements_ok += 1
            else:
                self.elements_scrap += 1
                print(f"  [{self.environ.now:7.2f}] {elem} → SCRAP")

        return good_elements


class AssemblyStation:
    """
    Assembly station with a single machine (SimPy Resource).
    Assembly time ~ Uniform(10, 20) min.
    Requires 1 superior cover + 1 inferior cover + 1 interior element.
    """

    def __init__(self, environ: simpy.Environment, capacity: int = 1):
        self.environ = environ
        self.resource = simpy.Resource(environ, capacity=capacity)
        self.products_assembled: int = 0
        self.stats = StationStats()

    def assemble(self, tapa_sup: Cover, tapa_inf: Cover,
                 elem_int: InteriorElement) -> FinalProduct:
        """SimPy process: assemble one finished product."""
        with self.resource.request() as req:
            entity_id = id(tapa_sup)
            self.stats.record_queue_entry(self.environ.now, entity_id)
            yield req
            self.stats.record_service_start(self.environ.now, entity_id)
            assembly_time = random.uniform(10, 20)
            yield self.environ.timeout(assembly_time)
            self.stats.record_service_end(self.environ.now)
            self.products_assembled += 1

        product = FinalProduct(
            id=self.products_assembled,
            superior_cover=tapa_sup,
            inferior_cover=tapa_inf,
            interior_element=elem_int,
            assembled_at=self.environ.now,
        )
        return product


# ---------------------------------------------------------------------------
# Model orchestrator
# ---------------------------------------------------------------------------

class AssemblyModel:
    """
    Wires the three flows together and runs the simulation.

    Buffers between stages are modelled with simpy.Store so that the
    assembly process blocks until one item of each type is available.
    """

    def __init__(self, environ: simpy.Environment, seed: int = 42):
        self.environ = environ
        random.seed(seed)

        # Stations / Resources
        self.paint = PaintStation(environ)
        self.unpack = UnpackingStation(environ)
        self.assembly = AssemblyStation(environ)

        # Buffers (inter-stage stores)
        self.queue_superior_cover: simpy.Store = simpy.Store(environ)
        self.queue_inferior_cover: simpy.Store = simpy.Store(environ)
        self.queue_interior_element: simpy.Store = simpy.Store(environ)

        # Buffer statistics
        self.stats_superior_cover = BufferStats()
        self.stats_inferior_cover = BufferStats()
        self.stats_interior_element = BufferStats()

        # Counters
        self.counter_cover: int = 0
        self.counter_box: int = 0
        self.finished_products: list[FinalProduct] = []

    # ----- generators (arrival processes) -----

    def arrival_cover(self):
        """Generates covers with Exp(11) inter-arrival, 50/50 sup/inf."""
        while True:
            inter_arrival = random.expovariate(1.0 / 11.0)
            yield self.environ.timeout(inter_arrival)

            self.counter_cover += 1
            cover_type = 'superior' if random.random() < 0.5 else 'inferior'
            cover = Cover(id=self.counter_cover, cover_type=cover_type,
                         created_at=self.environ.now)
            print(f"  [{self.environ.now:7.2f}] Arrival {cover}")

            # Launch the painting process for this cover
            self.environ.process(self.process_paint(cover))

    def arrival_box(self):
        """Generates boxes of 3 interior elements with Exp(64) inter-arrival."""
        while True:
            inter_arrival = random.expovariate(1.0 / 64.0)
            yield self.environ.timeout(inter_arrival)

            self.counter_box += 1
            box = InteriorElementsBox(id=self.counter_box,
                                        created_at=self.environ.now)
            print(f"  [{self.environ.now:7.2f}] Arrival {box}")

            # Launch the unpacking process for this box
            self.environ.process(self.process_unpacking(box))

    # ----- stage processes -----

    def process_paint(self, cover: Cover):
        """Paint a cover (with potential rework) and place it in the buffer."""
        yield self.environ.process(self.paint.paint(cover))

        print(f"  [{self.environ.now:7.2f}] {cover} painted OK "
              f"(attempts: {cover.paint_attempts})")

        if cover.cover_type == 'superior':
            yield self.queue_superior_cover.put(cover)
            self.stats_superior_cover.record_change(
                self.environ.now, len(self.queue_superior_cover.items))
        else:
            yield self.queue_inferior_cover.put(cover)
            self.stats_inferior_cover.record_change(
                self.environ.now, len(self.queue_inferior_cover.items))

    def process_unpacking(self, box: InteriorElementsBox):
        """Unpack a box and place good elements in the buffer."""
        good_elements = yield self.environ.process(
            self.unpack.unpack(box))

        for elem in good_elements:
            print(f"  [{self.environ.now:7.2f}] {elem} OK → buffer")
            yield self.queue_interior_element.put(elem)
            self.stats_interior_element.record_change(
                self.environ.now, len(self.queue_interior_element.items))

    def process_assembly(self):
        """
        Continuously wait for one of each component and assemble a product.
        Uses simpy.Store.get() which blocks until an item is available.
        """
        while True:
            # Wait for all three components (order does not matter,
            # but each get blocks independently)
            cover_sup = yield self.queue_superior_cover.get()
            self.stats_superior_cover.record_change(
                self.environ.now, len(self.queue_superior_cover.items))
            cover_inf = yield self.queue_inferior_cover.get()
            self.stats_inferior_cover.record_change(
                self.environ.now, len(self.queue_inferior_cover.items))
            interior_element = yield self.queue_interior_element.get()
            self.stats_interior_element.record_change(
                self.environ.now, len(self.queue_interior_element.items))

            print(f"  [{self.environ.now:7.2f}] Assembly started: "
                  f"{cover_sup}, {cover_inf}, {interior_element}")

            product = yield self.environ.process(self.assembly.assemble(cover_sup, cover_inf, interior_element))

            self.finished_products.append(product)
            print(f"  [{self.environ.now:7.2f}] ✓ {product}")

    # ----- simulation entry point -----

    def run(self, until: float = 1000.0):
        """Activate all processes and run the simulation."""
        # Start arrival generators
        self.environ.process(self.arrival_cover())
        self.environ.process(self.arrival_box())

        # Start the assembly consumer
        self.environ.process(self.process_assembly())

        # Run
        print(f"=== Assembly simulation – duration {until} min ===\n")
        self.environ.run(until=until)

        # Report
        self._print_report(until)

    def _print_report(self, sim_time: float):
        """Print summary statistics at the end of the simulation."""
        print("\n" + "=" * 60)
        print("SIMULATION SUMMARY")
        print("=" * 60)
        print(f"  Simulated time:                 {sim_time:.0f} min")
        print(f"  Covers generated:               {self.counter_cover}")
        print(f"  Covers painted (total ops):     "
              f"{self.paint.painted_covers}")
        print(f"  Covers reworked:                "
              f"{self.paint.reworked_covers}")
        print(f"  Boxes received:                 {self.counter_box}")
        print(f"  Boxes unpacked:                 "
              f"{self.unpack.boxes_unpacked}")
        print(f"  Interior elements OK:           "
              f"{self.unpack.elements_ok}")
        print(f"  Interior elements scrapped:     "
              f"{self.unpack.elements_scrap}")
        print(f"  Finished products:              "
              f"{len(self.finished_products)}")

        # Buffer levels at end of simulation
        print(f"\n  Buffer superior covers (final): "
              f"{len(self.queue_superior_cover.items)}")
        print(f"  Buffer inferior covers (final): "
              f"{len(self.queue_inferior_cover.items)}")
        print(f"  Buffer interior elements (final): "
              f"{len(self.queue_interior_element.items)}")

        # Resource utilization
        print(f"\n  {'RESOURCE UTILIZATION':}")
        print(f"  {'Paint station:':25s}"
              f"{self.paint.stats.utilization(sim_time) * 100:5.1f}%")
        print(f"  {'Unpacking station:':25s}"
              f"{self.unpack.stats.utilization(sim_time) * 100:5.1f}%")
        print(f"  {'Assembly station:':25s}"
              f"{self.assembly.stats.utilization(sim_time) * 100:5.1f}%")

        # Average queue waiting times
        print(f"\n  {'AVERAGE QUEUE WAITING TIMES (minutes)':}")
        print(f"  {'Paint queue:':25s}"
              f"{self.paint.stats.average_wait_time():7.2f}")
        print(f"  {'Unpacking queue:':25s}"
              f"{self.unpack.stats.average_wait_time():7.2f}")
        print(f"  {'Assembly queue:':25s}"
              f"{self.assembly.stats.average_wait_time():7.2f}")

        # Average queue lengths
        print(f"\n  {'AVERAGE QUEUE LENGTHS (entities)':}")
        print(f"  {'Paint queue:':25s}"
              f"{self.paint.stats.average_queue_length(sim_time):7.2f}")
        print(f"  {'Unpacking queue:':25s}"
              f"{self.unpack.stats.average_queue_length(sim_time):7.2f}")
        print(f"  {'Assembly queue:':25s}"
              f"{self.assembly.stats.average_queue_length(sim_time):7.2f}")

        # Buffer levels (average / final)
        print(f"\n  {'BUFFER LEVELS (average / final)':}")
        print(f"  {'Superior covers:':25s}"
              f"{self.stats_superior_cover.average_length(sim_time):5.2f} / "
              f"{len(self.queue_superior_cover.items)}")
        print(f"  {'Inferior covers:':25s}"
              f"{self.stats_inferior_cover.average_length(sim_time):5.2f} / "
              f"{len(self.queue_inferior_cover.items)}")
        print(f"  {'Interior elements:':25s}"
              f"{self.stats_interior_element.average_length(sim_time):5.2f} / "
              f"{len(self.queue_interior_element.items)}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env = simpy.Environment()
    model = AssemblyModel(env, seed=42)
    model.run(until=5000)