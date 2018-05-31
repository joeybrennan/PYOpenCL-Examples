
from time import time
import pyopencl as pycl
import numpy
import os

def generateET(wavelen,modes,AN,Z):
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    with open('generteEfield.cl', 'r') as myfile:
        integratePI = myfile.read()

    # Create context, queue and build program
    context = pycl.create_some_context()
    queue = pycl.CommandQueue(context)

    program = pycl.Program(context, integratePI).build()

    # Get the max work group size for the kernel pi on our device
    device = context.devices[0]

    steps = 1001
    h_x = (numpy.linspace(-30*wavelen,30*wavelen,steps).astype(numpy.float32))
    h_AN = (numpy.asarray(AN).astype(numpy.float32))


    
    d_x = pycl.Buffer(context, pycl.mem_flags.READ_ONLY | pycl.mem_flags.COPY_HOST_PTR, hostbuf=h_x)
    d_AN = pycl.Buffer(context, pycl.mem_flags.READ_ONLY | pycl.mem_flags.COPY_HOST_PTR, hostbuf=h_AN)

    # Start the timer
    rtime = time()
    z = numpy.linspace(10e-3,300e-3,301)
    z = numpy.append(z,Z)
    z = numpy.unique(z)
    ET = []
    ETz = []
    pi = program.pi
    pi.set_scalar_arg_dtypes([None, None,None,None,numpy.int32,numpy.int32,numpy.float32,numpy.float32])
    for zn in z:
        #ETz = []
        h_ETR = numpy.empty(steps).astype(numpy.float32)
        h_ETI = numpy.empty(steps).astype(numpy.float32)
        d_ETR = pycl.Buffer(context, pycl.mem_flags.WRITE_ONLY, h_ETR.nbytes)
        d_ETI = pycl.Buffer(context, pycl.mem_flags.WRITE_ONLY, h_ETI.nbytes)
        pi(queue, h_x.shape, None, d_x,d_AN,d_ETR,d_ETI,steps,modes,wavelen,zn)
        #print("done")
        
        pycl.enqueue_copy(queue, h_ETR, d_ETR)
        pycl.enqueue_copy(queue, h_ETI, d_ETI)
        ET.append(h_ETR+1j*h_ETI)
        if zn == Z:
            ETz = h_ETR+1j*h_ETI
        #ETT.append(ETz)
    # Stop the timer
    rtime = time() - rtime
    print(rtime)
    return ET,ETz,h_x,z


def generateAN(wavelen,modes):
    
    with open('generateAN.cl', 'r') as myfile:
        integratePI = myfile.read()

    # Some constant values
    INSTEPS = 512*512
    ITERS = 262144/2048

    # Set some default values:
    # Default number of steps (updated later to device prefereable)
    in_nsteps = INSTEPS
    # Default number of iterations
    niters = ITERS

    # Create context, queue and build program
    context = pycl.create_some_context()
    queue = pycl.CommandQueue(context)

    program = pycl.Program(context, integratePI).build()
    pi = program.pi
    pi.set_scalar_arg_dtypes([numpy.int32, numpy.int32,numpy.float32,numpy.float32,numpy.float32, None, None])

    # Get the max work group size for the kernel pi on our device
    device = context.devices[0]

    work_group_size = program.pi.get_work_group_info(pycl.kernel_work_group_info.WORK_GROUP_SIZE, device)
    

    # Now that we know the size of the work_groups, we can set the number of work
    # groups, the actual number of steps, and the step size
    nwork_groups = in_nsteps/(work_group_size*niters)
    print(nwork_groups)
    # if nwork_groups < 1:
    #     nwork_groups = device.max_compute_units
    #     work_group_size = in_nsteps/(nwork_groups*niters)

    nsteps = work_group_size * niters * nwork_groups

    d = 3*wavelen
    t = 6*wavelen

    
    #Define Bounds
    a1 = -d - t/2
    b2 = d + t/2
    
    start = a1
    end = b2
    step_size = (end-start) / float(nsteps)
    print(step_size)
    # # vector to hold partial sum
    h_psum = numpy.empty(int(nwork_groups)).astype(numpy.float32)

    print("%s work groups of size %s"  %( nwork_groups, work_group_size))
    print("Integration steps %s" %nsteps)

    d_partial_sums = pycl.Buffer(context, pycl.mem_flags.WRITE_ONLY, h_psum.nbytes)

    # Start the timer
    rtime = time()

    # # Execute the kernel over the entire range of our 1d input data et
    # # using the maximum number of work group items for this device
    # # Set the global and local size as tuples
    global_size = (int(nwork_groups * work_group_size),)
    local_size = ((work_group_size),)
    localmem = pycl.LocalMemory(numpy.dtype(numpy.float32).itemsize * work_group_size)
    print(niters)
    AN =[]
    for n in range(0,modes):
        pi(queue, global_size, local_size, int(n),int(niters), step_size, start,wavelen, localmem, d_partial_sums)
        #print("done")
        pycl.enqueue_copy(queue, h_psum, d_partial_sums)

        # # complete the sum and compute the final integral value
        pi_res = (h_psum.sum() * step_size)
        AN.append(pi_res)



        # Stop the timer
    rtime = time() - rtime
    print(rtime)
    return AN
    #print("The calculation ran in %s secs" %rtime)
    #print("pi = %s for %s steps" %(pi_res,nsteps))
    
    
    