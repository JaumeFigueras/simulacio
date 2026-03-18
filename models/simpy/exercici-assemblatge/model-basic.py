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
    """
    A cover piece (superior or inferior).

    Attributes
    ----------
    id : int
        Unique identifier for this cover.
    cover_type : str
        Type of cover: ``'superior'`` or ``'inferior'``.
    created_at : float
        Simulation time at which this cover was created.
    paint_attempts : int
        Number of times this cover has been painted (including rework).
    ready_at : float or None
        Simulation time at which this cover passed quality control,
        or ``None`` if not yet ready.
    queue_entry_time : float or None
        Simulation time at which this cover entered the inter-stage
        queue, or ``None`` if not yet queued.
    """

    id: int
    cover_type: str
    created_at: float = 0.0
    paint_attempts: int = 0
    ready_at: Optional[float] = None
    queue_entry_time: Optional[float] = None

    def __str__(self) -> str:
        """
        Return a human-readable string representation.

        Returns
        -------
        str
            Formatted string with id and cover type.
        """
        return f"Cover({self.id}, {self.cover_type})"


@dataclass
class InteriorElement:
    """
    An interior element unpacked from a box.

    Attributes
    ----------
    id : int
        Unique identifier for this interior element.
    box_id : int
        Identifier of the box this element was unpacked from.
    created_at : float
        Simulation time at which the source box arrived.
    ready_at : float or None
        Simulation time at which this element passed inspection,
        or ``None`` if not yet inspected or scrapped.
    queue_entry_time : float or None
        Simulation time at which this element entered the inter-stage
        queue, or ``None`` if not yet queued.
    """

    id: int
    box_id: int
    created_at: float = 0.0
    ready_at: Optional[float] = None
    queue_entry_time: Optional[float] = None

    def __str__(self) -> str:
        """
        Return a human-readable string representation.

        Returns
        -------
        str
            Formatted string with id and source box id.
        """
        return f"InteriorElement({self.id}, box={self.box_id})"


@dataclass
class InteriorElementsBox:
    """
    A box containing interior elements.

    Attributes
    ----------
    id : int
        Unique identifier for this box.
    created_at : float
        Simulation time at which this box arrived.
    num_elements : int
        Number of interior elements contained in this box.
    """

    id: int
    created_at: float = 0.0
    num_elements: int = 3

    def __str__(self) -> str:
        """
        Return a human-readable string representation.

        Returns
        -------
        str
            Formatted string with the box id.
        """
        return f"InteriorElementsBox({self.id})"


@dataclass
class FinalProduct:
    """
    A finished assembled product.

    Attributes
    ----------
    id : int
        Unique identifier for this finished product.
    superior_cover : Cover
        The superior cover used in this product.
    inferior_cover : Cover
        The inferior cover used in this product.
    interior_element : InteriorElement
        The interior element used in this product.
    assembled_at : float
        Simulation time at which the assembly was completed.
    """

    id: int
    superior_cover: Cover
    inferior_cover: Cover
    interior_element: InteriorElement
    assembled_at: float = 0.0

    def __str__(self) -> str:
        """
        Return a human-readable string representation.

        Returns
        -------
        str
            Formatted string with the product id and its components.
        """
        return (f"FinalProduct({self.id}: "
                f"{self.superior_cover}, {self.inferior_cover}, "
                f"{self.interior_element})")


# ---------------------------------------------------------------------------
# Statistics classes
# ---------------------------------------------------------------------------

class StationStats:
    """
    Tracks resource utilization, queue waiting times, and time-weighted
    average queue length for a single station (SimPy Resource).

    This class must be instrumented manually within SimPy process
    methods by calling its ``record_*`` methods at the appropriate
    points in the process flow.

    Attributes
    ----------
    total_busy_time : float
        Cumulative time the resource has been busy serving entities.
    total_wait_time : float
        Cumulative time entities have spent waiting in the queue.
    wait_count : int
        Total number of entities that have waited in the queue.
    current_queue_length : int
        Current number of entities waiting in the queue.
    max_queue_length : int
        Maximum observed queue length during the simulation.

    Examples
    --------
    Typical usage within a SimPy process::

        stats.record_queue_entry(env.now)     # before yield req
        yield req                             # wait for resource
        stats.record_service_start(env.now)   # service begins
        yield env.timeout(service_time)       # service
        stats.record_service_end(env.now)     # service ends
    """

    def __init__(self):
        """
        Initialize all statistics accumulators to zero.
        """
        # Resource utilization
        self.total_busy_time: float = 0.0
        self._busy_start: float = 0.0

        # Queue waiting times
        self.total_wait_time: float = 0.0
        self.wait_count: int = 0
        self._current_wait_start: float = 0.0

        # Time-weighted average queue length (area-under-the-curve)
        self.current_queue_length: int = 0
        self._last_queue_change_time: float = 0.0
        self._queue_length_area: float = 0.0
        self.max_queue_length: int = 0

    def record_queue_entry(self, now: float) -> None:
        """
        Record that an entity has joined the queue.

        Must be called **before** ``yield req``.

        Parameters
        ----------
        now : float
            Current simulation time.
        """
        self._current_wait_start = now
        self._queue_length_area += self.current_queue_length * (now - self._last_queue_change_time)
        self.current_queue_length += 1
        self._last_queue_change_time = now
        if self.current_queue_length > self.max_queue_length:
            self.max_queue_length = self.current_queue_length

    def record_service_start(self, now: float) -> None:
        """
        Record that an entity has started being served.

        Must be called **after** ``yield req`` (when the resource is granted).

        Parameters
        ----------
        now : float
            Current simulation time.
        """
        wait = now - self._current_wait_start
        self.total_wait_time += wait
        self.wait_count += 1

        self._queue_length_area += self.current_queue_length * (now - self._last_queue_change_time)
        self.current_queue_length -= 1
        self._last_queue_change_time = now

        self._busy_start = now

    def record_service_end(self, now: float) -> None:
        """
        Record that an entity has finished being served.

        Must be called after the service timeout completes.

        Parameters
        ----------
        now : float
            Current simulation time.
        """
        self.total_busy_time += now - self._busy_start

    def utilization(self, sim_time: float) -> float:
        """
        Compute resource utilization as a fraction.

        Parameters
        ----------
        sim_time : float
            Total simulation time.

        Returns
        -------
        float
            Utilization ratio in [0, 1].
        """
        if sim_time <= 0:
            return 0.0
        return self.total_busy_time / sim_time

    def avg_wait_time(self) -> float:
        """
        Compute the average time entities waited in the queue.

        Returns
        -------
        float
            Average wait time in minutes.  Returns 0.0 if no entities
            have been served yet.
        """
        if self.wait_count == 0:
            return 0.0
        return self.total_wait_time / self.wait_count

    def avg_queue_length(self, sim_time: float) -> float:
        """
        Compute the time-weighted average queue length.

        Uses the area-under-the-curve method, including the last
        interval up to ``sim_time``.

        Parameters
        ----------
        sim_time : float
            Total simulation time.

        Returns
        -------
        float
            Average number of entities in the queue.
        """
        if sim_time <= 0:
            return 0.0
        total_area = (self._queue_length_area
                      + self.current_queue_length * (sim_time - self._last_queue_change_time))
        return total_area / sim_time


class QueueStats:
    """
    Tracks time-weighted average length, maximum length, and average
    waiting time for a ``simpy.Store`` inter-stage queue.

    Call ``record_put`` when an item is added to the queue, and
    ``record_get`` when an item is removed.  Items must carry a
    ``queue_entry_time`` attribute that is set before ``record_put``
    is called.

    Attributes
    ----------
    current_length : int
        Current number of items in the queue.
    max_length : int
        Maximum observed queue length during the simulation.
    total_wait_time : float
        Cumulative time items have spent waiting in the queue.
    wait_count : int
        Total number of items that have been retrieved from the queue.
    """

    def __init__(self):
        """
        Initialize all statistics accumulators to zero.
        """
        # Time-weighted average queue length (area-under-the-curve)
        self.current_length: int = 0
        self._last_change_time: float = 0.0
        self._length_area: float = 0.0
        self.max_length: int = 0

        # Queue waiting times
        self.total_wait_time: float = 0.0
        self.wait_count: int = 0

    def record_put(self, now: float) -> None:
        """
        Record that an item has been added to the queue.

        The item's ``queue_entry_time`` attribute must be set to
        ``now`` **before** calling this method.

        Parameters
        ----------
        now : float
            Current simulation time.
        """
        self._length_area += self.current_length * (now - self._last_change_time)
        self.current_length += 1
        self._last_change_time = now
        if self.current_length > self.max_length:
            self.max_length = self.current_length

    def record_get(self, now: float, entry_time: float) -> None:
        """
        Record that an item has been removed from the queue.

        Parameters
        ----------
        now : float
            Current simulation time.
        entry_time : float
            The simulation time at which the item entered the queue
            (i.e., the item's ``queue_entry_time`` attribute).
        """
        # Update queue length area
        self._length_area += self.current_length * (now - self._last_change_time)
        self.current_length -= 1
        self._last_change_time = now

        # Update waiting time
        self.total_wait_time += now - entry_time
        self.wait_count += 1

    def avg_length(self, sim_time: float) -> float:
        """
        Compute the time-weighted average queue length.

        Parameters
        ----------
        sim_time : float
            Total simulation time.

        Returns
        -------
        float
            Average number of items in the queue.
        """
        if sim_time <= 0:
            return 0.0
        total_area = (self._length_area
                      + self.current_length * (sim_time - self._last_change_time))
        return total_area / sim_time

    def avg_wait_time(self) -> float:
        """
        Compute the average time items waited in the queue.

        Returns
        -------
        float
            Average wait time in minutes.  Returns 0.0 if no items
            have been retrieved yet.
        """
        if self.wait_count == 0:
            return 0.0
        return self.total_wait_time / self.wait_count


class WipStats:
    """
    Tracks time-weighted average Work in Progress (WIP) for a
    category of entities.

    WIP represents the number of entities currently in the system
    (from arrival to departure).  An entity enters WIP when it
    arrives and leaves WIP when it is consumed (assembled) or
    discarded (scrapped).

    Uses the area-under-the-curve method for time-weighted averaging.

    Attributes
    ----------
    current_wip : int
        Current number of entities in progress.
    max_wip : int
        Maximum observed WIP during the simulation.

    Examples
    --------
    Typical usage::

        wip.record_entry(env.now)   # entity arrives / enters the system
        # ... entity flows through stations ...
        wip.record_exit(env.now)    # entity is consumed or scrapped
    """

    def __init__(self):
        """
        Initialize all WIP accumulators to zero.
        """
        self.current_wip: int = 0
        self._last_change_time: float = 0.0
        self._wip_area: float = 0.0
        self.max_wip: int = 0

    def record_entry(self, now: float) -> None:
        """
        Record that an entity has entered the system (WIP increases).

        Parameters
        ----------
        now : float
            Current simulation time.
        """
        self._wip_area += self.current_wip * (now - self._last_change_time)
        self.current_wip += 1
        self._last_change_time = now
        if self.current_wip > self.max_wip:
            self.max_wip = self.current_wip

    def record_exit(self, now: float) -> None:
        """
        Record that an entity has left the system (WIP decreases).

        Called when an entity is consumed by assembly or discarded
        as scrap.

        Parameters
        ----------
        now : float
            Current simulation time.
        """
        self._wip_area += self.current_wip * (now - self._last_change_time)
        self.current_wip -= 1
        self._last_change_time = now

    def avg_wip(self, sim_time: float) -> float:
        """
        Compute the time-weighted average WIP.

        Parameters
        ----------
        sim_time : float
            Total simulation time.

        Returns
        -------
        float
            Average number of entities in progress.
        """
        if sim_time <= 0:
            return 0.0
        total_area = (self._wip_area
                      + self.current_wip * (sim_time - self._last_change_time))
        return total_area / sim_time


# ---------------------------------------------------------------------------
# Resource / Station classes
# ---------------------------------------------------------------------------

class PaintStation:
    """
    Painting station with a single server (SimPy Resource).

    Paint time follows a Uniform(6, 12) min distribution.
    After painting, 95% of covers pass QC; the remaining 5% return
    to the FIFO queue for rework.

    Parameters
    ----------
    environ : simpy.Environment
        The SimPy simulation environment.
    capacity : int, optional
        Number of parallel servers at this station (default is 1).

    Attributes
    ----------
    environ : simpy.Environment
        The SimPy simulation environment.
    resource : simpy.Resource
        The SimPy resource representing the painting server(s).
    painted_covers : int
        Total number of paint operations performed (including rework).
    reworked_covers : int
        Number of covers that failed QC and were sent back for rework.
    stats : StationStats
        Statistics tracker for utilization, wait times, and queue lengths.
    """

    def __init__(self, environ: simpy.Environment, capacity: int = 1):
        """
        Initialize the paint station.

        Parameters
        ----------
        environ : simpy.Environment
            The SimPy simulation environment.
        capacity : int, optional
            Number of parallel servers (default is 1).
        """
        self.environ = environ
        self.resource = simpy.Resource(environ, capacity=capacity)
        self.painted_covers: int = 0
        self.reworked_covers: int = 0
        self.stats = StationStats()

    def paint(self, cover: Cover):
        """
        SimPy process: paint a cover with potential rework loop.

        The cover enters the FIFO queue, is painted, and undergoes
        quality control.  If it fails QC (5% probability), it
        re-enters the queue for another painting attempt.

        Parameters
        ----------
        cover : Cover
            The cover entity to be painted.

        Yields
        ------
        simpy.events.Event
            SimPy events for resource requests and timeouts.
        """
        while True:
            cover.paint_attempts += 1

            # Record queue entry before requesting the resource
            self.stats.record_queue_entry(self.environ.now)

            with self.resource.request() as req:
                yield req

                # Record service start (queue wait ends, resource busy)
                self.stats.record_service_start(self.environ.now)

                paint_time = random.uniform(6, 12)
                yield self.environ.timeout(paint_time)

                # Record service end
                self.stats.record_service_end(self.environ.now)
                self.painted_covers += 1

            # Quality control
            if random.random() < 0.95:
                cover.ready_at = self.environ.now
                return  # exit the loop – cover is good
            else:
                self.reworked_covers += 1
                print(f"  [{self.environ.now:7.2f}] {cover} REWORK "
                      f"(attempt {cover.paint_attempts})")


class UnpackingStation:
    """
    Unpacking station with a single machine (SimPy Resource).

    Unpack time follows a Uniform(30, 50) min distribution per box
    of 3 elements.  During unpacking, 10% of interior elements are
    identified as defective and discarded as scrap.

    Parameters
    ----------
    environ : simpy.Environment
        The SimPy simulation environment.
    capacity : int, optional
        Number of parallel machines at this station (default is 1).

    Attributes
    ----------
    environ : simpy.Environment
        The SimPy simulation environment.
    resource : simpy.Resource
        The SimPy resource representing the unpacking machine(s).
    boxes_unpacked : int
        Total number of boxes that have been unpacked.
    elements_ok : int
        Number of interior elements that passed inspection.
    elements_scrap : int
        Number of interior elements discarded as scrap.
    stats : StationStats
        Statistics tracker for utilization, wait times, and queue lengths.
    """

    def __init__(self, environ: simpy.Environment, capacity: int = 1):
        """
        Initialize the unpacking station.

        Parameters
        ----------
        environ : simpy.Environment
            The SimPy simulation environment.
        capacity : int, optional
            Number of parallel machines (default is 1).
        """
        self.environ = environ
        self.resource = simpy.Resource(environ, capacity=capacity)
        self.boxes_unpacked: int = 0
        self.elements_ok: int = 0
        self.elements_scrap: int = 0
        self.stats = StationStats()

    def unpack(self, box: InteriorElementsBox):
        """
        SimPy process: unpack a box and inspect its elements.

        The box waits in the FIFO queue for the machine, is unpacked,
        and each element is inspected.  Defective elements (10%) are
        discarded.

        Parameters
        ----------
        box : InteriorElementsBox
            The box entity to be unpacked.

        Yields
        ------
        simpy.events.Event
            SimPy events for resource requests and timeouts.

        Returns
        -------
        list of InteriorElement
            List of interior elements that passed inspection.
        """
        # Record queue entry
        self.stats.record_queue_entry(self.environ.now)

        with self.resource.request() as req:
            yield req

            # Record service start
            self.stats.record_service_start(self.environ.now)

            unpack_time = random.uniform(30, 50)
            yield self.environ.timeout(unpack_time)

            # Record service end
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

    Assembly time follows a Uniform(10, 20) min distribution.
    Requires 1 superior cover + 1 inferior cover + 1 interior element.

    Parameters
    ----------
    environ : simpy.Environment
        The SimPy simulation environment.
    capacity : int, optional
        Number of parallel machines at this station (default is 1).

    Attributes
    ----------
    environ : simpy.Environment
        The SimPy simulation environment.
    resource : simpy.Resource
        The SimPy resource representing the assembly machine(s).
    products_assembled : int
        Total number of products that have been assembled.
    stats : StationStats
        Statistics tracker for utilization, wait times, and queue lengths.
    """

    def __init__(self, environ: simpy.Environment, capacity: int = 1):
        """
        Initialize the assembly station.

        Parameters
        ----------
        environ : simpy.Environment
            The SimPy simulation environment.
        capacity : int, optional
            Number of parallel machines (default is 1).
        """
        self.environ = environ
        self.resource = simpy.Resource(environ, capacity=capacity)
        self.products_assembled: int = 0
        self.stats = StationStats()

    def assemble(self, cover_sup: Cover, cover_inf: Cover,
                 elem_int: InteriorElement) -> FinalProduct:
        """
        SimPy process: assemble one finished product.

        The three components wait in the FIFO queue for the assembly
        machine, then are assembled into a final product.

        Parameters
        ----------
        cover_sup : Cover
            The superior cover component.
        cover_inf : Cover
            The inferior cover component.
        elem_int : InteriorElement
            The interior element component.

        Yields
        ------
        simpy.events.Event
            SimPy events for resource requests and timeouts.

        Returns
        -------
        FinalProduct
            The assembled final product.
        """
        # Record queue entry
        self.stats.record_queue_entry(self.environ.now)

        with self.resource.request() as req:
            yield req

            # Record service start
            self.stats.record_service_start(self.environ.now)

            assembly_time = random.uniform(10, 20)
            yield self.environ.timeout(assembly_time)

            # Record service end
            self.stats.record_service_end(self.environ.now)
            self.products_assembled += 1

        product = FinalProduct(
            id=self.products_assembled,
            superior_cover=cover_sup,
            inferior_cover=cover_inf,
            interior_element=elem_int,
            assembled_at=self.environ.now,
        )
        return product


# ---------------------------------------------------------------------------
# Model orchestrator
# ---------------------------------------------------------------------------

class AssemblyModel:
    """
    Orchestrates the three production flows and runs the simulation.

    Inter-stage queues are modelled with ``simpy.Store`` so that the
    assembly process blocks until one item of each type is available.

    Parameters
    ----------
    environ : simpy.Environment
        The SimPy simulation environment.
    seed : int, optional
        Random seed for reproducibility (default is 42).
    verbose : bool, optional
        If ``True``, print trace messages during the simulation
        (default is ``True``).

    Attributes
    ----------
    environ : simpy.Environment
        The SimPy simulation environment.
    verbose : bool
        Whether trace messages are printed during the simulation.
    paint : PaintStation
        The painting station.
    unpack : UnpackingStation
        The unpacking station.
    assembly : AssemblyStation
        The assembly station.
    queue_superior_cover : simpy.Store
        Queue holding painted superior covers awaiting assembly.
    queue_inferior_cover : simpy.Store
        Queue holding painted inferior covers awaiting assembly.
    queue_interior_element : simpy.Store
        Queue holding inspected interior elements awaiting assembly.
    stats_queue_sup : QueueStats
        Statistics tracker for the superior cover queue.
    stats_queue_inf : QueueStats
        Statistics tracker for the inferior cover queue.
    stats_queue_elem : QueueStats
        Statistics tracker for the interior element queue.
    wip_covers : WipStats
        WIP tracker for cover entities.
    wip_elements : WipStats
        WIP tracker for interior element entities.
    wip_total : WipStats
        WIP tracker for all entities combined.
    counter_cover : int
        Total number of covers generated.
    counter_box : int
        Total number of boxes generated.
    finished_products : list of FinalProduct
        List of all assembled final products.
    """

    def __init__(self, environ: simpy.Environment, seed: int = 42,
                 verbose: bool = True):
        """
        Initialize the assembly model.

        Parameters
        ----------
        environ : simpy.Environment
            The SimPy simulation environment.
        seed : int, optional
            Random seed for reproducibility (default is 42).
        verbose : bool, optional
            If ``True``, print trace messages (default is ``True``).
        """
        self.environ = environ
        self.verbose = verbose
        random.seed(seed)

        # Stations / Resources
        self.paint = PaintStation(environ)
        self.unpack = UnpackingStation(environ)
        self.assembly = AssemblyStation(environ)

        # Inter-stage queues
        self.queue_superior_cover: simpy.Store = simpy.Store(environ)
        self.queue_inferior_cover: simpy.Store = simpy.Store(environ)
        self.queue_interior_element: simpy.Store = simpy.Store(environ)

        # Queue statistics
        self.stats_queue_sup = QueueStats()
        self.stats_queue_inf = QueueStats()
        self.stats_queue_elem = QueueStats()

        # WIP statistics
        self.wip_covers = WipStats()
        self.wip_elements = WipStats()
        self.wip_total = WipStats()

        # Counters
        self.counter_cover: int = 0
        self.counter_box: int = 0
        self.finished_products: list[FinalProduct] = []

        # Simulation time (set after run)
        self._sim_time: float = 0.0

    def _log(self, msg: str) -> None:
        """
        Print a trace message if verbose mode is enabled.

        Parameters
        ----------
        msg : str
            The message to print.
        """
        if self.verbose:
            print(msg)

    # ----- generators (arrival processes) -----

    def arrival_cover(self):
        """
        Generate covers with exponential inter-arrival times.

        Covers arrive following an Exp(11 min) distribution.
        Each cover is randomly assigned as ``'superior'`` or
        ``'inferior'`` with equal probability (50/50).

        Yields
        ------
        simpy.events.Event
            SimPy timeout events for inter-arrival delays.
        """
        while True:
            inter_arrival = random.expovariate(1.0 / 11.0)
            yield self.environ.timeout(inter_arrival)

            self.counter_cover += 1
            cover_type = 'superior' if random.random() < 0.5 else 'inferior'
            cover = Cover(id=self.counter_cover, cover_type=cover_type,
                         created_at=self.environ.now)
            self._log(f"  [{self.environ.now:7.2f}] Arrival {cover}")

            # WIP: cover enters the system
            self.wip_covers.record_entry(self.environ.now)
            self.wip_total.record_entry(self.environ.now)

            # Launch the painting process for this cover
            self.environ.process(self.process_paint(cover))

    def arrival_box(self):
        """
        Generate boxes of interior elements with exponential inter-arrival times.

        Boxes arrive following an Exp(64 min) distribution.
        Each box contains 3 interior elements.

        Yields
        ------
        simpy.events.Event
            SimPy timeout events for inter-arrival delays.
        """
        while True:
            inter_arrival = random.expovariate(1.0 / 64.0)
            yield self.environ.timeout(inter_arrival)

            self.counter_box += 1
            box = InteriorElementsBox(id=self.counter_box,
                                        created_at=self.environ.now)
            self._log(f"  [{self.environ.now:7.2f}] Arrival {box}")

            # WIP: all elements in the box enter the system
            for _ in range(box.num_elements):
                self.wip_elements.record_entry(self.environ.now)
                self.wip_total.record_entry(self.environ.now)

            # Launch the unpacking process for this box
            self.environ.process(self.process_unpacking(box))

    # ----- stage processes -----

    def process_paint(self, cover: Cover):
        """
        Paint a cover and place it in the appropriate queue.

        Delegates painting (with potential rework) to the
        ``PaintStation``, then routes the finished cover to the
        superior or inferior cover queue.

        Parameters
        ----------
        cover : Cover
            The cover entity to be painted.

        Yields
        ------
        simpy.events.Event
            SimPy process events from the paint station.
        """
        yield self.environ.process(self.paint.paint(cover))

        self._log(f"  [{self.environ.now:7.2f}] {cover} painted OK "
                  f"(attempts: {cover.paint_attempts})")

        # Stamp queue entry time on the entity and update queue stats
        cover.queue_entry_time = self.environ.now

        if cover.cover_type == 'superior':
            self.queue_superior_cover.put(cover)
            self.stats_queue_sup.record_put(self.environ.now)
        else:
            self.queue_inferior_cover.put(cover)
            self.stats_queue_inf.record_put(self.environ.now)

    def process_unpacking(self, box: InteriorElementsBox):
        """
        Unpack a box and place good elements in the interior element queue.

        Delegates unpacking and inspection to the ``UnpackingStation``,
        then places each element that passed inspection into the queue.
        Scrapped elements are recorded as WIP exits.

        Parameters
        ----------
        box : InteriorElementsBox
            The box entity to be unpacked.

        Yields
        ------
        simpy.events.Event
            SimPy process events from the unpacking station.
        """
        good_elements = yield self.environ.process(
            self.unpack.unpack(box))

        # Count scrapped elements and record WIP exits for them
        num_scrapped = box.num_elements - len(good_elements)
        for _ in range(num_scrapped):
            self.wip_elements.record_exit(self.environ.now)
            self.wip_total.record_exit(self.environ.now)

        for elem in good_elements:
            self._log(f"  [{self.environ.now:7.2f}] {elem} OK → queue")
            # Stamp queue entry time on the entity and update queue stats
            elem.queue_entry_time = self.environ.now
            self.queue_interior_element.put(elem)
            self.stats_queue_elem.record_put(self.environ.now)

    def process_assembly(self):
        """
        Continuously assemble products from queued components.

        Waits for one superior cover, one inferior cover, and one
        interior element to become available in their respective
        queues, then delegates assembly to the ``AssemblyStation``.
        Upon completion, all three components exit WIP.
        Uses ``simpy.Store.get()`` which blocks until an item is
        available.

        Yields
        ------
        simpy.events.Event
            SimPy store get events and process events.
        """
        while True:
            # Wait for all three components and record queue retrievals
            cover_sup = yield self.queue_superior_cover.get()
            self.stats_queue_sup.record_get(
                self.environ.now, cover_sup.queue_entry_time)

            cover_inf = yield self.queue_inferior_cover.get()
            self.stats_queue_inf.record_get(
                self.environ.now, cover_inf.queue_entry_time)

            interior_element = yield self.queue_interior_element.get()
            self.stats_queue_elem.record_get(
                self.environ.now, interior_element.queue_entry_time)

            self._log(f"  [{self.environ.now:7.2f}] Assembly started: "
                      f"{cover_sup}, {cover_inf}, {interior_element}")

            product = yield self.environ.process(
                self.assembly.assemble(cover_sup, cover_inf, interior_element))

            self.finished_products.append(product)
            self._log(f"  [{self.environ.now:7.2f}] ✓ {product}")

            # WIP: all three components leave the system
            self.wip_covers.record_exit(self.environ.now)   # superior
            self.wip_covers.record_exit(self.environ.now)   # inferior
            self.wip_elements.record_exit(self.environ.now)  # interior
            self.wip_total.record_exit(self.environ.now)     # sup cover
            self.wip_total.record_exit(self.environ.now)     # inf cover
            self.wip_total.record_exit(self.environ.now)     # int element

    # ----- results extraction -----

    def get_results(self) -> dict:
        """
        Extract all output statistics as a flat dictionary.

        This method is designed to be called after ``run()`` completes.
        The returned dictionary can be collected across multiple
        replications for Monte Carlo analysis.

        Returns
        -------
        dict
            Dictionary with the following keys:

            **Parts counters**

            - ``'covers_generated'`` : int
            - ``'covers_painted'`` : int
            - ``'covers_reworked'`` : int
            - ``'boxes_received'`` : int
            - ``'boxes_unpacked'`` : int
            - ``'elements_ok'`` : int
            - ``'elements_scrapped'`` : int
            - ``'finished_products'`` : int

            **Resource utilization** (fraction 0–1)

            - ``'util_paint'`` : float
            - ``'util_unpack'`` : float
            - ``'util_assembly'`` : float

            **Station queue average waiting times** (minutes)

            - ``'wait_paint'`` : float
            - ``'wait_unpack'`` : float
            - ``'wait_assembly'`` : float

            **Station queue average lengths** (entities)

            - ``'qlen_paint'`` : float
            - ``'qlen_unpack'`` : float
            - ``'qlen_assembly'`` : float

            **Inter-stage queue average waiting times** (minutes)

            - ``'wait_sup_covers'`` : float
            - ``'wait_inf_covers'`` : float
            - ``'wait_int_elements'`` : float

            **Inter-stage queue average lengths** (entities)

            - ``'qlen_sup_covers'`` : float
            - ``'qlen_inf_covers'`` : float
            - ``'qlen_int_elements'`` : float

            **Work in Progress** (time-weighted average entities)

            - ``'wip_covers'`` : float
            - ``'wip_elements'`` : float
            - ``'wip_total'`` : float
        """
        t = self._sim_time
        return {
            # Parts counters
            'covers_generated': self.counter_cover,
            'covers_painted': self.paint.painted_covers,
            'covers_reworked': self.paint.reworked_covers,
            'boxes_received': self.counter_box,
            'boxes_unpacked': self.unpack.boxes_unpacked,
            'elements_ok': self.unpack.elements_ok,
            'elements_scrapped': self.unpack.elements_scrap,
            'finished_products': len(self.finished_products),
            # Resource utilization
            'util_paint': self.paint.stats.utilization(t),
            'util_unpack': self.unpack.stats.utilization(t),
            'util_assembly': self.assembly.stats.utilization(t),
            # Station queue average waiting times
            'wait_paint': self.paint.stats.avg_wait_time(),
            'wait_unpack': self.unpack.stats.avg_wait_time(),
            'wait_assembly': self.assembly.stats.avg_wait_time(),
            # Station queue average lengths
            'qlen_paint': self.paint.stats.avg_queue_length(t),
            'qlen_unpack': self.unpack.stats.avg_queue_length(t),
            'qlen_assembly': self.assembly.stats.avg_queue_length(t),
            # Inter-stage queue average waiting times
            'wait_sup_covers': self.stats_queue_sup.avg_wait_time(),
            'wait_inf_covers': self.stats_queue_inf.avg_wait_time(),
            'wait_int_elements': self.stats_queue_elem.avg_wait_time(),
            # Inter-stage queue average lengths
            'qlen_sup_covers': self.stats_queue_sup.avg_length(t),
            'qlen_inf_covers': self.stats_queue_inf.avg_length(t),
            'qlen_int_elements': self.stats_queue_elem.avg_length(t),
            # Work in Progress
            'wip_covers': self.wip_covers.avg_wip(t),
            'wip_elements': self.wip_elements.avg_wip(t),
            'wip_total': self.wip_total.avg_wip(t),
        }

    # ----- simulation entry point -----

    def run(self, until: float = 1000.0):
        """
        Activate all processes and run the simulation.

        Starts the cover and box arrival generators, the assembly
        consumer, and runs the SimPy environment until the specified
        time.  Prints a summary report at the end if verbose mode is
        enabled.

        Parameters
        ----------
        until : float, optional
            Simulation end time in minutes (default is 1000.0).
        """
        self._sim_time = until

        # Start arrival generators
        self.environ.process(self.arrival_cover())
        self.environ.process(self.arrival_box())

        # Start the assembly consumer
        self.environ.process(self.process_assembly())

        # Run
        self._log(f"=== Assembly simulation – duration {until} min ===\n")
        self.environ.run(until=until)

        # Report
        if self.verbose:
            self._print_report(until)

    def _print_report(self, sim_time: float):
        """
        Print summary statistics at the end of the simulation.

        Displays production counters, resource utilization, average
        queue waiting times, average queue lengths (with maximums),
        inter-stage queue levels and waiting times, and WIP statistics.

        Parameters
        ----------
        sim_time : float
            Total simulation time in minutes.
        """
        print("\n" + "=" * 60)
        print("SIMULATION SUMMARY")
        print("=" * 60)

        # --- Production counters ---
        print("\n  PRODUCTION COUNTERS")
        print("  " + "-" * 56)
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

        # --- Resource utilization ---
        print("\n  RESOURCE UTILIZATION")
        print("  " + "-" * 56)
        print(f"  Paint station:                  "
              f"{self.paint.stats.utilization(sim_time) * 100:.1f}%")
        print(f"  Unpacking station:              "
              f"{self.unpack.stats.utilization(sim_time) * 100:.1f}%")
        print(f"  Assembly station:               "
              f"{self.assembly.stats.utilization(sim_time) * 100:.1f}%")

        # --- Station queue average waiting times ---
        print("\n  STATION QUEUE AVERAGE WAITING TIMES (minutes)")
        print("  " + "-" * 56)
        print(f"  Paint queue:                    "
              f"{self.paint.stats.avg_wait_time():.2f}")
        print(f"  Unpacking queue:                "
              f"{self.unpack.stats.avg_wait_time():.2f}")
        print(f"  Assembly queue:                 "
              f"{self.assembly.stats.avg_wait_time():.2f}")

        # --- Station queue average lengths ---
        print("\n  STATION QUEUE AVERAGE LENGTHS (entities)")
        print("  " + "-" * 56)
        print(f"  Paint queue:                    "
              f"{self.paint.stats.avg_queue_length(sim_time):.2f}"
              f"  (max: {self.paint.stats.max_queue_length})")
        print(f"  Unpacking queue:                "
              f"{self.unpack.stats.avg_queue_length(sim_time):.2f}"
              f"  (max: {self.unpack.stats.max_queue_length})")
        print(f"  Assembly queue:                 "
              f"{self.assembly.stats.avg_queue_length(sim_time):.2f}"
              f"  (max: {self.assembly.stats.max_queue_length})")

        # --- Inter-stage queue waiting times ---
        print("\n  INTER-STAGE QUEUE AVERAGE WAITING TIMES (minutes)")
        print("  " + "-" * 56)
        print(f"  Superior covers:                "
              f"{self.stats_queue_sup.avg_wait_time():.2f}")
        print(f"  Inferior covers:                "
              f"{self.stats_queue_inf.avg_wait_time():.2f}")
        print(f"  Interior elements:              "
              f"{self.stats_queue_elem.avg_wait_time():.2f}")

        # --- Inter-stage queue levels ---
        print("\n  INTER-STAGE QUEUE LEVELS (avg / max / final)")
        print("  " + "-" * 56)
        print(f"  Superior covers:                "
              f"{self.stats_queue_sup.avg_length(sim_time):.2f}"
              f" / {self.stats_queue_sup.max_length}"
              f" / {len(self.queue_superior_cover.items)}")
        print(f"  Inferior covers:                "
              f"{self.stats_queue_inf.avg_length(sim_time):.2f}"
              f" / {self.stats_queue_inf.max_length}"
              f" / {len(self.queue_inferior_cover.items)}")
        print(f"  Interior elements:              "
              f"{self.stats_queue_elem.avg_length(sim_time):.2f}"
              f" / {self.stats_queue_elem.max_length}"
              f" / {len(self.queue_interior_element.items)}")

        # --- Work in Progress ---
        print("\n  WORK IN PROGRESS (time-weighted average / max / current)")
        print("  " + "-" * 56)
        print(f"  Covers WIP:                     "
              f"{self.wip_covers.avg_wip(sim_time):.2f}"
              f" / {self.wip_covers.max_wip}"
              f" / {self.wip_covers.current_wip}")
        print(f"  Interior elements WIP:          "
              f"{self.wip_elements.avg_wip(sim_time):.2f}"
              f" / {self.wip_elements.max_wip}"
              f" / {self.wip_elements.current_wip}")
        print(f"  Total WIP:                      "
              f"{self.wip_total.avg_wip(sim_time):.2f}"
              f" / {self.wip_total.max_wip}"
              f" / {self.wip_total.current_wip}")

        print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env = simpy.Environment()
    model = AssemblyModel(env, seed=42)
    model.run(until=5000)