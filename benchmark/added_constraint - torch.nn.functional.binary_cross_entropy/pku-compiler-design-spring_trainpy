import os
import pickle
import tvm
import torch
import torch.multiprocessing as _multi
multi = _multi.get_context("spawn")
import numpy as np
import logging
import heapq
import time
from auto_schedule.examples import FUNC_TABLE
from auto_schedule.scheduler import graph_analysis, op_schedule_cpu_general_dx, able_inline, op_schedule_gpu_general_dx
from auto_schedule.measure import serial_evaluate, batch_evaluate, _evaluate
from auto_schedule.models import OpScheduleCPUd5, OpScheduleGPUd5
from auto_schedule.test import test_graph_schedule_gpu_general_dx
from auto_schedule.utils import to_tuple


MAX_CPU = 20
MAX_GPU = 3
C1 = 1
C2 = 1
LR = 0.002
MT = 0.7


class Entity(object):
    def __init__(self, func_name, args):
        self.func_name = func_name
        self.args = args


def train_op_schedule_cpu_general_dx(entities, epoch, batch_size, path, loop_num=100, loop_size=16,
                                     stack_size=20, logfile="temp.log", device="cuda:0"):
    dim = 5
    timeout = 15.0
    num_sample = len(entities)
    device = torch.device(device)
    model = OpScheduleCPUd5(3, 128, device)
    # load or initialize parameter file
    if os.path.exists(path) and os.path.isfile(path):
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
    else:
        torch.save(model.state_dict(), path)
    model.to(device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=LR)
    model.train()
    # maintain a dataset for each function
    datasets = [[] for i in range(num_sample)]

    train_beg_time = time.time()
    with open(logfile, "a") as f:
        f.write("New log\ntime: {}".format(train_beg_time))
    perf_before = dict()
    perf_before_dump = False
    model.train()
    print("Scheduling begins...parameters in path {}\n    logs to{}".format(path, logfile))
    for i in range(epoch):
        optimizer.zero_grad()
        for batch in range(batch_size):
            for p in range(num_sample):
                func_name = entities[p].func_name
                func = FUNC_TABLE[func_name].func
                args = entities[p].args
                ops, bufs = func(*args)
                s = tvm.create_schedule(ops)
                # get the performance before scheduling
                # only run one time
                entity_key = "{}:{}".format(func_name, args)
                if entity_key not in perf_before:
                    pre_cost = serial_evaluate(s, bufs, "llvm", np.random.randint(0, MAX_CPU), 10, timeout=timeout)
                    perf_before[entity_key] = pre_cost
                if not isinstance(ops, (list, tuple)):
                    ops = [ops]
                bfs_order, down_graph = graph_analysis(ops)
                group_points = []
                for op in bfs_order:
                    if not isinstance(op, tvm.tensor.ComputeOp):
                        continue
                    if able_inline(op, down_graph):
                        s[op].compute_inline()
                    else:
                        group_points.append(op)
                if len(group_points) > 1:
                    raise RuntimeError("Not support more than one compute")
                for j, point in enumerate(group_points):
                    y_dict, y_diary = op_schedule_cpu_general_dx(dim, s, point, model, random=np.random.random() < 0.2, sampling=True)
                    post_cost = serial_evaluate(s, bufs, "llvm", np.random.randint(0, MAX_CPU), 10, timeout=timeout)
                    data = dict()
                    for name, value in y_dict.items():
                        if isinstance(value, list):
                            tmp = []
                            for v in value:
                                tmp.append(v.detach())
                            data[name] = (tmp, y_diary[name])   # the data record schedule decisions
                        else:
                            data[name] = (value.detach(), y_diary[name])
                        # record  (point No. , sch data, time cost)
                        datasets[p].append((j, data, post_cost))
        # record performance before scheduling
        # only run one time
        if not perf_before_dump:
            with open(logfile, "a") as f:
                logs = "performance before scheduling:\n"
                f.write(logs)
                for key, perf in perf_before.items():
                    logs = "{}: {}\n".format(key, perf)
                    f.write(logs)
                f.write("\n")
            perf_before_dump = True
        # control the size of dataset and record best cases
        cur_time = time.time()
        with open(logfile, "a") as f:
            for j in range(num_sample):
                datasets[j] = heapq.nsmallest(stack_size, datasets[j], key=lambda x: x[-1])
                entity_key = "{}:{}".format(entities[j].func_name, entities[j].args)
                duration = cur_time - train_beg_time
                logs = "epoch {}/{}| {} best perf {}| [{}s]\n".format(i+1, epoch, entity_key, datasets[j][0][-1], duration)
                f.write(logs)
                logs = "schedule {}\n".format(entity_key)
                for name, val in datasets[j][0][1].items():    # find the diary, this is ugly now, change later
                    logs = logs + "{}: {}\n".format(name, val[1])
                logs = logs + "\n"
                f.write(logs)
        # train the parameters
        for r in range(loop_num):
            acc_loss = 0.0
            for inner in range(loop_size):
                for q in range(num_sample):
                    func_name = entities[q].func_name
                    func = FUNC_TABLE[func_name].func
                    args = entities[q].args
                    for (point_num, data, time_cost) in datasets[q][:1]:
                        ops, bufs = func(*args)
                        s = tvm.create_schedule(ops)
                        if not isinstance(ops, (list, tuple)):
                            ops = [ops]
                        bfs_order, down_graph = graph_analysis(ops)
                        group_points = []
                        for op in bfs_order:
                            if not isinstance(op, tvm.tensor.ComputeOp):
                                continue
                            if able_inline(op, down_graph):
                                s[op].compute_inline()
                            else:
                                group_points.append(op)
                        y_dict, _ = op_schedule_cpu_general_dx(dim, s, group_points[point_num], model, random=False, sampling=False)
                        # spatial loss
                        spatial_loss = 0.0
                        for j in range(dim):
                            spatial_loss = spatial_loss + torch.nn.functional\
                                .binary_cross_entropy(y_dict["spatial"][j], data["spatial"][0][j])
                        # reduce_loss
                        reduce_loss = 0.0
                        for j in range(dim):
                            reduce_loss = reduce_loss + torch.nn.functional\
                                .binary_cross_entropy(y_dict["reduce"][j], data["reduce"][0][j])
                        # parallel_loss
                        parallel_loss = torch.nn.functional\
                            .binary_cross_entropy(y_dict["parallel"], data["parallel"][0])
                        # reorder_one loss
                        reorder_one_loss = torch.nn.functional\
                            .binary_cross_entropy(y_dict["reorder_one"], data["reorder_one"][0])
                        # reorder_two loss
                        reorder_two_loss = torch.nn.functional\
                            .binary_cross_entropy(y_dict["reorder_two"], data["reorder_two"][0])
                        # reorder_three loss
                        reorder_three_loss = torch.nn.functional\
                            .binary_cross_entropy(y_dict["reorder_three"], data["reorder_three"][0])
                        # accumulate loss
                        acc_loss = acc_loss + spatial_loss + reduce_loss + parallel_loss + reorder_one_loss \
                                   + reorder_two_loss + reorder_three_loss
            acc_loss.backward()
            if r % 10 == 0:
                torch.save(model.state_dict(), path)
                logs = "epoch={}, r={}, loss={}\n".format(i + 1, r, float(acc_loss.detach()))
                with open(logfile, "a") as f:
                    f.write(logs)
            optimizer.step()
        with open(logfile, "a") as f:
            f.write("\n")
    print("All done.")


def train_op_schedule_gpu_general_dx(entities, epoch, batch_size, path, loop_num=100, loop_size=32,
                                     stack_size=20, logfile="temp.log", device="cuda:0"):
    dim = 5
    timeout = 15.0
    max_trial = 100
    num_entity = len(entities)
    datasets = [[] for i in range(num_entity)]
    # initialize parameter file
    if not os.path.exists(path):
        model = OpScheduleGPUd5(3, 128)
        torch.save(model.state_dict(), path)
    train_beg_time = time.time()
    print("Schedule begins...parameters in {}\n    logs to {}".format(path, logfile), flush=True)
    with open(logfile, "a") as f:
        f.write("New log [{}]\n".format(train_beg_time))
    for ep_num in range(epoch):
        for batch in range(batch_size):
            for p_entity in range(num_entity):
                trial = 0
                while trial < max_trial:
                    entity = entities[p_entity]
                    queue = multi.Queue()
                    proc = multi.Process(target=_get_data_gpu, args=(dim, entity, path, queue, device))
                    proc.start()
                    proc.join(timeout=timeout)
                    proc.terminate()
                    proc.join()
                    if not queue.empty():
                        data_file_path = queue.get(block=True)
                        if os.path.exists(data_file_path):
                            with open(data_file_path, "rb") as f:
                                data_lst = pickle.load(f)
                            for count, (data, time_cost) in enumerate(data_lst):
                                datasets[p_entity].append((count, data, time_cost))
                            os.remove(data_file_path)
                    trial += 1
                    if len(datasets[p_entity]) > 0:
                        break
        # control dataset
        cur_time = time.time()
        with open(logfile, "a") as f:
            for j in range(num_entity):
                datasets[j] = heapq.nsmallest(stack_size, datasets[j], key=lambda x: x[-1])
                entity_key = "{}:{}".format(entities[j].func_name, entities[j].args)
                duration = cur_time - train_beg_time
                logs = "epoch {}/{}| {} best perf {}| [{}s]\n".format(ep_num + 1, epoch, entity_key, datasets[j][0][-1],
                                                                      duration)
                f.write(logs)
                logs = "schedule {}\n".format(entity_key)
                for name, val in datasets[j][0][1].items():  # find the diary, this is ugly now, change later
                    logs = logs + "{}: {}\n".format(name, val[1])
                logs = logs + "\n"
                f.write(logs)
        # train on dataset
        real_device = torch.device(device)
        model = OpScheduleGPUd5(3, 128, real_device)
        model.to(real_device)
        if os.path.exists(path):
            state_dict = torch.load(path)
            model.load_state_dict(state_dict)
        model.train()
        optimizer = torch.optim.Adadelta(model.parameters(), lr=LR)
        _train_gpu(dim, entities, model, optimizer, datasets, loop_num, loop_size, logfile, real_device)
        torch.save(model.state_dict(), path)
    print("All done.")


def _get_data_gpu(dim, entity, model_path, queue, device_str):
    func = FUNC_TABLE[entity.func_name].func
    args = entity.args
    ops, bufs = func(*args)
    device = torch.device(device_str)
    model = OpScheduleGPUd5(3, 128)
    if os.path.exists(model_path):
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
    model.train()
    s = tvm.create_schedule(ops)
    if not isinstance(ops, (list, tuple)):
        ops = [ops]
    bfs_order, down_graph = graph_analysis(ops)
    group_points = []
    for op in bfs_order:
        if not isinstance(op, tvm.tensor.ComputeOp):
            continue
        if able_inline(op, down_graph):
            s[op].compute_inline()
        else:
            group_points.append(op)
    if len(group_points) > 1:   # do not support more than one compute
        return
    data_lst = []
    for count, point in enumerate(group_points):
        y_dict, diary = op_schedule_gpu_general_dx(dim, s, point, model, random=np.random.random() < 0.5, sampling=True)
        try:
            time_cost = _evaluate(s, bufs, "cuda", dev_id=np.random.randint(0, MAX_GPU), number=1)
            data = dict()
            for name, value in y_dict.items():
                if isinstance(value, list):
                    tmp = []
                    for v in value:
                        tmp.append(v.detach().tolist())
                    data[name] = (tmp, diary[name])
                else:
                    data[name] = (value.detach(), diary[name])
            data_lst.append((data, time_cost))
        except Exception as e:
            pass
    # TODO random string
    file_path = "tmp.txt"
    with open(file_path, "wb") as f:
        pickle.dump(data_lst, f)
    queue.put(file_path)


def _train_gpu(dim, entities, model, optimizer, datasets, loop_num, loop_size, logfile, device):
    for r in range(loop_num):
        acc_loss = 0.0
        for inner in range(loop_size):
            for q in range(len(entities)):
                for (count, data, time_cost) in datasets[q][:1]:
                    func_name = entities[q].func_name
                    func = FUNC_TABLE[func_name].func
                    args = entities[q].args
                    ops, bufs = func(*args)
                    s = tvm.create_schedule(ops)
                    if not isinstance(ops, (list, tuple)):
                        ops = [ops]
                    bfs_order, down_graph = graph_analysis(ops)
                    group_points = []
                    for op in bfs_order:
                        if not isinstance(op, tvm.tensor.ComputeOp):
                            continue
                        if able_inline(op, down_graph):
                            s[op].compute_inline()
                        else:
                            group_points.append(op)
                    y_dict, diary = op_schedule_gpu_general_dx(dim, s, group_points[count], model, random=False, sampling=False)
                    # spatial loss one
                    spatial_loss = 0.0
                    # spatial part one
                    for i in range(dim):
                        spatial_loss = spatial_loss + torch.nn.functional\
                            .binary_cross_entropy(y_dict["spatial_one"][i], torch.FloatTensor(data["spatial_one"][0][i]).to(device))
                    # spatial part three
                    for i in range(dim):
                        if diary["spatial_one"][i] == data["spatial_one"][1][i]:
                            spatial_loss = spatial_loss + torch.nn.functional\
                                .binary_cross_entropy(y_dict["spatial_three"][i], torch.FloatTensor(data["spatial_three"][0][i]).to(device))
                    # reduce_loss
                    reduce_loss = 0.0
                    for i in range(dim):
                        reduce_loss = reduce_loss + torch.nn.functional\
                            .binary_cross_entropy(y_dict["reduce"][i], torch.FloatTensor(data["reduce"][0][i]).to(device))
                    # reorder_one loss
                    reorder_one_loss = torch.nn.functional\
                        .binary_cross_entropy(y_dict["reorder_one"], torch.FloatTensor(data["reorder_one"][0]).to(device))
                    # reorder_two loss
                    reorder_two_loss = torch.nn.functional\
                        .binary_cross_entropy(y_dict["reorder_two"], torch.FloatTensor(data["reorder_two"][0]).to(device))
                    # reorder_three loss
                    reorder_three_loss = torch.nn.functional\
                        .binary_cross_entropy(y_dict["reorder_three"], torch.FloatTensor(data["reorder_three"][0]).to(device))
                    # accumulate loss
                    acc_loss = acc_loss + spatial_loss + reduce_loss + reorder_one_loss + reorder_two_loss + reorder_three_loss
        acc_loss.backward()
        if r % 10 == 0:
            logs = "loss={}\n".format(float(acc_loss.detach()))
            with open(logfile, "a") as f:
                f.write(logs)
        optimizer.step()
    with open(logfile, "a") as f:
        f.write("\n")


def _eval_gpu(dim, entity, model_path, queue, trial=10, number=10):
    func_name = entity.func_name
    func = FUNC_TABLE[func_name].func
    args = entity.args
    best_time = float("+inf")
    model = OpScheduleGPUd5(3, 128)
    if os.path.exists(model_path):
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
    model.eval()
    for i in range(trial):
        ops, bufs = func(*args)
        s = tvm.create_schedule(ops)
        if not isinstance(ops, (list, tuple)):
            ops = [ops]
        bfs_order, down_graph = graph_analysis(ops)
        group_points = []
        for op in bfs_order:
            if not isinstance(op, tvm.tensor.ComputeOp):
                continue
            if able_inline(op, down_graph):
                s[op].compute_inline()
            else:
                group_points.append(op)
        op_schedule_gpu_general_dx(dim, s, group_points[0], model, random=False, sampling=True)
        try:
            time_cost = _evaluate(s, bufs, "cuda", dev_id=np.random.randint(0, MAX_GPU), number=number)
            if time_cost < best_time:
                best_time = time_cost
        except Exception as e:
            pass
    queue.put(best_time)


if __name__ == "__main__":
    entities = []
    # func = FUNC_TABLE["conv2d_channel_batch"].func
    # args = (1, 14, 14, 256, 3, 3, 512, 1, 1)
    # entities.append(Entity("conv2d_channel_batch", args))
    func = FUNC_TABLE["matmul_batch"].func
    args = (1, 1024, 1024, 1024)
    entities.append(Entity("matmul_batch", args))
    beg = time.time()
    train_op_schedule_gpu_general_dx(entities, 5, 16, "models/test_gemm_gpu.pkl")
    end = time.time()
    print("train done! use {}ms".format((end - beg) * 1e3))
    test_graph_schedule_gpu_general_dx(entities, "./models/test_gemm_gpu.pkl", sampling=True)
