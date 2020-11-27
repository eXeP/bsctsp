import os
import sys
from subprocess import Popen, PIPE
import csv

def run_and_get_input(cmd):
    print("running: " + cmd)
    process = Popen(cmd.split(" "), stdout=PIPE)
    (output, err) = process.communicate()
    exit_code = process.wait()
    res = str(output).replace("\\n", " ")
    print("res: " + res)
    return res

instancesG = ["eil101", "a280", "att532", "pr1002", "d1291", "d1655", "d2103", "pcb3038", "rl5915", "rl11849"]
instancesC = ["eil101", "a280", "att532", "pr1002", "d1291", "d1655", "d2103", "pcb3038"]
devices = ["GPU", "CPU"]

def run_rand(instance):
    out = run_and_get_input("./solver " + instance + " RAND GPU")
    outs = out.split(" ")
    result = outs[3]
    time = outs[1]
    return int(result), int(time)

def run_alpha(instance, alpha, device):
    out = run_and_get_input("./solver " + instance + " " + alpha + " " + device + " 10")
    outs = out.split(" ")
    return int(outs[5]), int(outs[3]), int(outs[1])

opt = [629, 2579, 27686, 259045, 50801, 62128, 80450, 137694, 565530, 923307]
with open('results_cpu.csv', 'w', newline='') as res_gpu, open('err_cpu.cpu', 'w', newline='') as err_gpu, open('time_cpu.csv', 'w', newline='') as time_gpu, open('time2_cpu.csv', 'w', newline='') as time2_gpu:
    res_writer = csv.writer(res_gpu)
    err_writer = csv.writer(err_gpu)
    time_writer = csv.writer(time_gpu)
    time2_writer = csv.writer(time2_gpu)
    res_writer.writerow(["Instance", "RAND result", "A1 result", "A2 result", "A3 result", "Optimum"])
    err_writer.writerow(["Instance", "RAND result", "A1 error", "A2 error", "A3 error"])
    time_writer.writerow(["Instance", "RAND time", "A1 time1", "A1 time2", "A2 time1", "A2 time2", "A3 time1", "A3 time2"])
    time2_writer.writerow(["Instance", "RAND time", "A1 time", "A2 time", "A3 time"])
    for i in range(0, len(instancesG)):
        instance = instancesG[i]
        RAND_res, RAND_time = run_rand(instance)
        A1_res, A1_t1, A1_t2 = run_alpha(instance, "ALPHA1", devices[0])
        A2_res, A2_t1, A2_t2 = run_alpha(instance, "ALPHA2", devices[0])
        A3_res, A3_t1, A3_t2 = run_alpha(instance, "ALPHA3", devices[0])
        res_writer.writerow([instance, RAND_res, A1_res, A2_res, A3_res, opt[i]])
        err_writer.writerow([instance, str(round((RAND_res-opt[i])/opt[i]*100, 1)), str(round((A1_res-opt[i])/opt[i]*100, 1)), str(round((A2_res-opt[i])/opt[i]*100, 1)), str(round((A3_res-opt[i])/opt[i]*100, 1))])
        time_writer.writerow([instance, RAND_time, A1_t1, A1_t2, A2_t1, A2_t2, A3_t1, A3_t2])
        time2_writer.writerow([instance, RAND_time, A1_t1+A1_t2, A2_t1+A2_t2, A3_t1+A3_t2])
        print(instance + " done")