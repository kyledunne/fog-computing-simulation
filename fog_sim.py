import numpy, queue, copy, random
from abc import ABC, abstractmethod


global_time = 0


def main():
    sim_input = SimulationInput(num_fog_nodes=10,
                                num_tasks=1500,
                                iot_to_fog_delay=10,
                                cloud_to_iot_delay=50,
                                fog_to_cloud_delay=40,
                                intra_fog_delay=4,
                                threshold=100,
                                fog_processing_speed=1,
                                cloud_processing_speed=1,
                                avg_task_size=10,  # time taken to process a task (in ms): task_size * processing_speed
                                stdev_task_size=3,
                                avg_tasks_created_per_tick=-5,
                                stdev_tasks_created_per_tick=3,
                                avg_deadline_margin=100,
                                stdev_deadline_margin=50,
                                offloading_max_jumps=5)
    networkOffloading = NetworkFogOffloading(sim_input.num_fog_nodes, sim_input.offloading_max_jumps)
    networkFIFO = Network1CommandNodeFIFO(sim_input.num_fog_nodes)
    networkEDD = Network1CommandNodeEDD(sim_input.num_fog_nodes)
    networkCR = Network1CommandNodeCR(sim_input.num_fog_nodes)
    simOffloading = Simulation(copy.deepcopy(sim_input), networkOffloading)
    simFIFO = Simulation(copy.deepcopy(sim_input), networkFIFO)
    simEDD = Simulation(copy.deepcopy(sim_input), networkEDD)
    simCR = Simulation(copy.deepcopy(sim_input), networkCR)
    simOffloading.simulate()
    simFIFO.simulate()
    simEDD.simulate()
    simCR.simulate()
    print('Offloading:')
    for result in simOffloading.results:
        print(result)
    print('FIFO:')
    for result in simFIFO.results:
        print(result)
    print('EDD:')
    for result in simEDD.results:
        print(result)
    print('CR:')
    for result in simCR.results:
        print(result)
    print()
    print('[Task size | Time created | Deadline | Time completed | Cloud/Fog]')
    print()
    print('Analytics:')
    print('Offloading:')
    analytics(simOffloading.results)
    print('FIFO:')
    analytics(simFIFO.results)
    print('EDD:')
    analytics(simEDD.results)
    print('CR:')
    analytics(simCR.results)


def analytics(results):
    # 1. Total system delay
    # 2. Number tardy tasks
    # 3. Number of tasks sent to cloud
    total_delay = 0
    for result in results:
        total_delay += result[3] - result[1]
    print('Total time to complete all tasks:', total_delay, 'ms')
    num_tardy = 0
    for result in results:
        if result[3] > result[2]:
            num_tardy += 1
    print('Number of tardy tasks:', num_tardy, 'out of', len(results))
    num_cloud = 0
    for result in results:
        if result[4] == 'c':
            num_cloud += 1
    print('Number of tasks sent to cloud:', num_cloud, 'out of', len(results))


class SimulationInput:
    def __init__(self, num_fog_nodes,
                 num_tasks,
                 iot_to_fog_delay,
                 cloud_to_iot_delay,
                 fog_to_cloud_delay,
                 intra_fog_delay,
                 threshold,
                 fog_processing_speed,
                 cloud_processing_speed,
                 avg_task_size,
                 stdev_task_size,
                 avg_tasks_created_per_tick,
                 stdev_tasks_created_per_tick,
                 avg_deadline_margin,
                 stdev_deadline_margin,
                 offloading_max_jumps):
        # generates a blank (all zeroes) matrix with the dimensions width = 4, height = num_tasks
        self.tasks = [[0 for x in range(4)] for y in range(num_tasks)]
        # sets each task to a random size, normally distributed around avg_task_size
        for x in range(num_tasks):
            self.tasks[x][0] = numpy.random.normal(loc=avg_task_size, scale=stdev_task_size, size=None)
        # sets the time of creation of each task. The way I did this is: each "tick" (millisecond) some number of tasks get created.
        # The number of tasks created per tick is random, and normally distributed around avg_tasks_created_per_tick
        # then, the time is randomly adjusted for each task up to 0.5 ms in either direction
        x = 0
        current_tick = 0
        while x < num_tasks:
            tasks_this_tick = numpy.random.normal(loc=avg_tasks_created_per_tick, scale=stdev_tasks_created_per_tick, size=None)
            y = 0
            while y < tasks_this_tick:
                if x >= num_tasks:
                    break
                self.tasks[x][1] = current_tick + random.uniform(-0.5, 0.5)
                x += 1
                y += 1
            current_tick += 1
        # Sets each task's deadline to its creation time + a randomly generated deadline margin
        for x in range(num_tasks):
            self.tasks[x][2] = self.tasks[x][1] + numpy.random.normal(loc=avg_deadline_margin, scale=stdev_deadline_margin, size=None)

        self.num_fog_nodes = num_fog_nodes
        self.iot_to_fog_delay = iot_to_fog_delay
        self.cloud_to_iot_delay = cloud_to_iot_delay
        self.fog_to_cloud_delay = fog_to_cloud_delay
        self.intra_fog_delay = intra_fog_delay
        self.threshold = threshold
        self.fog_processing_speed = fog_processing_speed
        self.cloud_processing_speed = cloud_processing_speed
        self.offloading_max_jumps = offloading_max_jumps


class Simulation:
    def __init__(self, sim_input, network):
        self.sim_input = sim_input
        self.time = 0
        self.evlist = queue.PriorityQueue()
        self.network = network
        self.results = []
        if not network.is_cr():
            for task in sim_input.tasks:
                self.evlist.put_nowait(Event(time=task[1], event_type='task_creation', data=Task(task)))
        else:
            for task in sim_input.tasks:
                self.evlist.put_nowait(Event(time=task[1], event_type='task_creation', data=CRTask(task)))

    def simulate(self):
        while True:
            try:
                current_event = self.evlist.get_nowait()
                self.time = current_event.time
                self.network.update_state(current_event, self)
            except queue.Empty:
                return self.conclude()

    def conclude(self):
        print("concluding")


class Network(ABC):
    @abstractmethod
    def update_state(self, event, simulation):
        pass

    @abstractmethod
    def is_cr(self):
        return False


class Network1CommandNodeFIFO(Network):
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.waiting_times = [0 for _ in range(num_nodes - 1)]
        self.exec_node_queues = [queue.Queue() for _ in range(num_nodes - 1)]
        self.is_processing = [False for _ in range(num_nodes-1)]

    def update_state(self, event, simulation):
        if event.type == 'task_creation':
            simulation.evlist.put_nowait(Event(time=event.time + simulation.sim_input.iot_to_fog_delay,
                                               event_type='arrival_at_command_node', data=event.data))
        elif event.type == 'arrival_at_command_node':
            #decide whether to send to cloud or an execution node
            choice_index = -1
            choice_queue_space = -1
            for exec_node in range(simulation.sim_input.num_fog_nodes - 1):
                if self.waiting_times[exec_node] < simulation.sim_input.threshold:
                    diff = simulation.sim_input.threshold - self.waiting_times[exec_node]
                    if diff > choice_queue_space:
                        choice_index = exec_node
                        choice_queue_space = diff
            #choice_index is remaining at -1 when it shouldn't
            # the waiting time at the exec_nodes is not going down?
            if choice_index == -1:
                simulation.evlist.put_nowait(Event(time=event.time + simulation.sim_input.fog_to_cloud_delay,
                                                   event_type='arrival_at_cloud', data=event.data))
            else:
                simulation.evlist.put_nowait(Event(time=event.time + simulation.sim_input.intra_fog_delay,
                                                   event_type='arrival_at_execution_node', data=[choice_index, event.data]))
                self.waiting_times[choice_index] += event.data.size * simulation.sim_input.fog_processing_speed

        elif event.type == 'arrival_at_execution_node':
            #update waiting time at command node
            #if no event is being processed, start processing
            if self.exec_node_queues[event.data[0]].empty():
                if not self.is_processing[event.data[0]]:
                    #creates execution_completed_fog event at currenttime + (task_size * fog_processing_speed)
                    simulation.evlist.put_nowait(Event(time=event.time + event.data[1].size * simulation.sim_input.fog_processing_speed,
                                                       event_type='execution_completed_fog', data=event.data))
                    self.is_processing[event.data[0]] = True
                    self.waiting_times[event.data[0]] -= event.data[1].size * simulation.sim_input.fog_processing_speed
                else:
                    #add this task to this node's queue
                    self.exec_node_queues[event.data[0]].put_nowait(event.data[1])
            else:
                #add this task to this node's queue
                self.exec_node_queues[event.data[0]].put_nowait(event.data[1])
        elif event.type == 'arrival_at_cloud':
            simulation.evlist.put_nowait(Event(time=event.time + event.data.size * simulation.sim_input.cloud_processing_speed,
                                               event_type='execution_completed_cloud', data=event.data))
        elif event.type == 'execution_completed_fog':
            #create returned_to_iot event
            #create execution_completed_fog event for next task in queue
            #if nothing in queue set is processing to false
            simulation.evlist.put_nowait(Event(time=event.time + simulation.sim_input.iot_to_fog_delay,
                                               event_type='returned_to_iot', data=[event.data[1], 'f']))
            if self.exec_node_queues[event.data[0]].empty():
                self.is_processing[event.data[0]] = False
            else:
                next_task = self.exec_node_queues[event.data[0]].get_nowait()
                simulation.evlist.put_nowait(Event(time=event.time + next_task.size * simulation.sim_input.fog_processing_speed,
                                                   event_type='execution_completed_fog', data=[event.data[0], next_task]))
                self.is_processing[event.data[0]] = True
                self.waiting_times[event.data[0]] -= next_task.size * simulation.sim_input.fog_processing_speed
        elif event.type == 'execution_completed_cloud':
            simulation.evlist.put_nowait(Event(time=event.time + simulation.sim_input.cloud_to_iot_delay,
                                               event_type='returned_to_iot', data=[event.data, 'c']))
        elif event.type == 'returned_to_iot':
            simulation.results.append([event.data[0].size, event.data[0].time_created, event.data[0].deadline, event.time, event.data[1]])
        else:
            print('wait what happened? event type is', event.type)
            exit()

    def is_cr(self):
        return False


class Network1CommandNodeEDD(Network):
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.waiting_times = [0 for _ in range(num_nodes - 1)]
        self.exec_node_queues = [queue.PriorityQueue() for _ in range(num_nodes - 1)]
        self.is_processing = [False for _ in range(num_nodes-1)]

    def update_state(self, event, simulation):
        if event.type == 'task_creation':
            simulation.evlist.put_nowait(Event(time=event.time + simulation.sim_input.iot_to_fog_delay,
                                               event_type='arrival_at_command_node', data=event.data))
        elif event.type == 'arrival_at_command_node':
            #decide whether to send to cloud or an execution node
            choice_index = -1
            choice_queue_space = -1
            for exec_node in range(simulation.sim_input.num_fog_nodes - 1):
                if self.waiting_times[exec_node] < simulation.sim_input.threshold:
                    diff = simulation.sim_input.threshold - self.waiting_times[exec_node]
                    if diff > choice_queue_space:
                        choice_index = exec_node
                        choice_queue_space = diff
            if choice_index == -1:
                simulation.evlist.put_nowait(Event(time=event.time + simulation.sim_input.fog_to_cloud_delay,
                                                   event_type='arrival_at_cloud', data=event.data))
            else:
                simulation.evlist.put_nowait(Event(time=event.time + simulation.sim_input.intra_fog_delay,
                                                   event_type='arrival_at_execution_node', data=[choice_index, event.data]))
                self.waiting_times[choice_index] += event.data.size * simulation.sim_input.fog_processing_speed

        elif event.type == 'arrival_at_execution_node':
            #update waiting time at command node
            #if no event is being processed, start processing
            if self.exec_node_queues[event.data[0]].empty():
                if not self.is_processing[event.data[0]]:
                    #creates execution_completed_fog event at currenttime + (task_size * fog_processing_speed)
                    simulation.evlist.put_nowait(Event(time=event.time + event.data[1].size * simulation.sim_input.fog_processing_speed,
                                                       event_type='execution_completed_fog', data=event.data))
                    self.is_processing[event.data[0]] = True
                    self.waiting_times[event.data[0]] -= event.data[1].size * simulation.sim_input.fog_processing_speed
                else:
                    #add this task to this node's queue
                    self.exec_node_queues[event.data[0]].put_nowait(event.data[1])
            else:
                #add this task to this node's queue
                self.exec_node_queues[event.data[0]].put_nowait(event.data[1])
        elif event.type == 'arrival_at_cloud':
            simulation.evlist.put_nowait(Event(time=event.time + event.data.size * simulation.sim_input.cloud_processing_speed,
                                               event_type='execution_completed_cloud', data=event.data))
        elif event.type == 'execution_completed_fog':
            #create returned_to_iot event
            #create execution_completed_fog event for next task in queue
            #if nothing in queue set is processing to false
            simulation.evlist.put_nowait(Event(time=event.time + simulation.sim_input.iot_to_fog_delay,
                                               event_type='returned_to_iot', data=[event.data[1], 'f']))
            if self.exec_node_queues[event.data[0]].empty():
                self.is_processing[event.data[0]] = False
            else:
                next_task = self.exec_node_queues[event.data[0]].get_nowait()
                simulation.evlist.put_nowait(Event(time=event.time + next_task.size * simulation.sim_input.fog_processing_speed,
                                                   event_type='execution_completed_fog', data=[event.data[0], next_task]))
                self.is_processing[event.data[0]] = True
                self.waiting_times[event.data[0]] -= next_task.size * simulation.sim_input.fog_processing_speed
        elif event.type == 'execution_completed_cloud':
            simulation.evlist.put_nowait(Event(time=event.time + simulation.sim_input.cloud_to_iot_delay,
                                               event_type='returned_to_iot', data=[event.data, 'c']))
        elif event.type == 'returned_to_iot':
            simulation.results.append([event.data[0].size, event.data[0].time_created, event.data[0].deadline, event.time, event.data[1]])
        else:
            print('wait what happened? event type is', event.type)
            exit()

    def is_cr(self):
        return False


class NetworkFogOffloading(Network):
    def __init__(self, num_nodes, maximum_jumps):
        self.num_nodes = num_nodes
        self.waiting_times = [0 for _ in range(num_nodes)]
        self.node_queues = [queue.Queue() for _ in range(num_nodes)]
        self.is_processing = [False for _ in range(num_nodes)]
        self.maximum_jumps = maximum_jumps

    def update_state(self, event, simulation):
        if event.type == 'task_creation':
            simulation.evlist.put_nowait(Event(time=event.time + simulation.sim_input.iot_to_fog_delay,
                                               event_type='arrival_at_fog_node', data=[random.randrange(0, self.num_nodes), event.data, 0]))
        elif event.type == 'arrival_at_fog_node':
            current_node = event.data[0]
            current_task = event.data[1]
            current_num_jumps = event.data[2]
            if self.waiting_times[current_node] < simulation.sim_input.threshold:
                if self.node_queues[current_node].empty():
                    if self.is_processing[current_node]:
                        self.node_queues[current_node].put_nowait(current_task)
                        #print('should equal 0:', self.waiting_times[current_node] == 0)
                        self.waiting_times[current_node] += current_task.size * simulation.sim_input.fog_processing_speed
                    else:
                        simulation.evlist.put_nowait(Event(time=event.time + current_task.size * simulation.sim_input.fog_processing_speed,
                                                           event_type='execution_completed_fog', data=[current_node, current_task]))
                        self.is_processing[current_node] = True
                else:
                    self.node_queues[current_node].put_nowait(current_task)
                    self.waiting_times[current_node] += current_task.size * simulation.sim_input.fog_processing_speed
            else:
                if current_num_jumps < self.maximum_jumps:
                    #offload
                    rand_next = random.randrange(0, self.num_nodes - 1)
                    if rand_next >= current_node:
                        rand_next += 1
                    simulation.evlist.put_nowait(Event(time=event.time + simulation.sim_input.intra_fog_delay,
                                                       event_type='arrival_at_fog_node', data=[current_node, current_task, current_num_jumps + 1]))
                else:
                    #send to cloud
                    simulation.evlist.put_nowait(Event(time=event.time + simulation.sim_input.fog_to_cloud_delay,
                                                       event_type='arrival_at_cloud', data=current_task))
        elif event.type == 'execution_completed_fog':
            current_node = event.data[0]
            current_task = event.data[1]
            simulation.evlist.put_nowait(Event(time=event.time + simulation.sim_input.iot_to_fog_delay,
                                               event_type='returned_to_iot', data=[current_task, 'f']))
            if self.node_queues[current_node].empty():
                self.is_processing[current_node] = False
            else:
                next_task = self.node_queues[current_node].get_nowait()
                self.waiting_times[current_node] -= next_task.size * simulation.sim_input.fog_processing_speed
                self.is_processing[current_node] = True
                simulation.evlist.put_nowait(Event(time=event.time + next_task.size * simulation.sim_input.fog_processing_speed,
                                                   event_type='execution_completed_fog', data=[current_node, next_task]))
        elif event.type == 'arrival_at_cloud':
            simulation.evlist.put_nowait(Event(time=event.time + event.data.size * simulation.sim_input.cloud_processing_speed,
                                               event_type='execution_completed_cloud', data=event.data))
        elif event.type == 'execution_completed_cloud':
            simulation.evlist.put_nowait(Event(time=event.time + simulation.sim_input.cloud_to_iot_delay,
                                               event_type='returned_to_iot', data=[event.data, 'c']))
        elif event.type == 'returned_to_iot':
            simulation.results.append([event.data[0].size, event.data[0].time_created, event.data[0].deadline, event.time, event.data[1]])
        else:
            print("wrong event type (Offloading Network):", event.type)
            exit()

    def is_cr(self):
        return False


class Network1CommandNodeCR(Network):
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.waiting_times = [0 for _ in range(num_nodes - 1)]
        self.exec_node_queues = [queue.PriorityQueue() for _ in range(num_nodes - 1)]
        self.is_processing = [False for _ in range(num_nodes-1)]

    def update_state(self, event, simulation):
        global global_time
        global_time = event.time
        if event.type == 'task_creation':
            simulation.evlist.put_nowait(Event(time=event.time + simulation.sim_input.iot_to_fog_delay,
                                               event_type='arrival_at_command_node', data=event.data))
        elif event.type == 'arrival_at_command_node':
            #decide whether to send to cloud or an execution node
            choice_index = -1
            choice_queue_space = -1
            for exec_node in range(simulation.sim_input.num_fog_nodes - 1):
                if self.waiting_times[exec_node] < simulation.sim_input.threshold:
                    diff = simulation.sim_input.threshold - self.waiting_times[exec_node]
                    if diff > choice_queue_space:
                        choice_index = exec_node
                        choice_queue_space = diff
            if choice_index == -1:
                simulation.evlist.put_nowait(Event(time=event.time + simulation.sim_input.fog_to_cloud_delay,
                                                   event_type='arrival_at_cloud', data=event.data))
            else:
                simulation.evlist.put_nowait(Event(time=event.time + simulation.sim_input.intra_fog_delay,
                                                   event_type='arrival_at_execution_node', data=[choice_index, event.data]))
                self.waiting_times[choice_index] += event.data.size * simulation.sim_input.fog_processing_speed

        elif event.type == 'arrival_at_execution_node':
            #update waiting time at command node
            #if no event is being processed, start processing
            if self.exec_node_queues[event.data[0]].empty():
                if not self.is_processing[event.data[0]]:
                    #creates execution_completed_fog event at currenttime + (task_size * fog_processing_speed)
                    simulation.evlist.put_nowait(Event(time=event.time + event.data[1].size * simulation.sim_input.fog_processing_speed,
                                                       event_type='execution_completed_fog', data=event.data))
                    self.is_processing[event.data[0]] = True
                    self.waiting_times[event.data[0]] -= event.data[1].size * simulation.sim_input.fog_processing_speed
                else:
                    #add this task to this node's queue
                    self.exec_node_queues[event.data[0]].put_nowait(event.data[1])
            else:
                #add this task to this node's queue
                self.exec_node_queues[event.data[0]].put_nowait(event.data[1])
        elif event.type == 'arrival_at_cloud':
            simulation.evlist.put_nowait(Event(time=event.time + event.data.size * simulation.sim_input.cloud_processing_speed,
                                               event_type='execution_completed_cloud', data=event.data))
        elif event.type == 'execution_completed_fog':
            #create returned_to_iot event
            #create execution_completed_fog event for next task in queue
            #if nothing in queue set is processing to false
            simulation.evlist.put_nowait(Event(time=event.time + simulation.sim_input.iot_to_fog_delay,
                                               event_type='returned_to_iot', data=[event.data[1], 'f']))
            if self.exec_node_queues[event.data[0]].empty():
                self.is_processing[event.data[0]] = False
            else:
                next_task = self.exec_node_queues[event.data[0]].get_nowait()
                simulation.evlist.put_nowait(Event(time=event.time + next_task.size * simulation.sim_input.fog_processing_speed,
                                                   event_type='execution_completed_fog', data=[event.data[0], next_task]))
                self.is_processing[event.data[0]] = True
                self.waiting_times[event.data[0]] -= next_task.size * simulation.sim_input.fog_processing_speed
        elif event.type == 'execution_completed_cloud':
            simulation.evlist.put_nowait(Event(time=event.time + simulation.sim_input.cloud_to_iot_delay,
                                               event_type='returned_to_iot', data=[event.data, 'c']))
        elif event.type == 'returned_to_iot':
            simulation.results.append([event.data[0].size, event.data[0].time_created, event.data[0].deadline, event.time, event.data[1]])
        else:
            print('wait what happened? event type is', event.type)
            exit()

    def is_cr(self):
        return True


class Task:
    def __init__(self, info):
        self.size = info[0]
        self.time_created = info[1]
        self.deadline = info[2]
        self.time_completed = info[3]

    def __eq__(self, other):
        return self.deadline == other.deadline

    def __lt__(self, other):
        return self.deadline < other.deadline

    def __gt__(self, other):
        return self.deadline > other.deadline


#processing time / time remaining until deadline
#use a global_time variable and compare based on that
class CRTask:
    def __init__(self, info):
        self.size = info[0]
        self.time_created = info[1]
        self.deadline = info[2]
        self.time_completed = info[3]

    def __eq__(self, other):
        first_cr = self.size / (self.deadline - global_time)
        second_cr = other.size / (other.deadline - global_time)
        return first_cr == second_cr

    def __lt__(self, other):
        first_cr = self.size / (self.deadline - global_time)
        second_cr = other.size / (other.deadline - global_time)
        return first_cr > second_cr

    def __gt__(self, other):
        first_cr = self.size / (self.deadline - global_time)
        second_cr = other.size / (other.deadline - global_time)
        return first_cr < second_cr


class Event:
    def __init__(self, time, event_type, data):
        self.time = time
        self.type = event_type
        self.data = data

    def __eq__(self, other):
        return self.time == other.time

    def __lt__(self, other):
        return self.time < other.time

    def __gt__(self, other):
        return self.time > other.time


if __name__ == '__main__':
    main()
