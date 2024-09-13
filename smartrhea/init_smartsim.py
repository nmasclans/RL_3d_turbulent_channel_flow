#!/usr/bin/env python3

"""
Module to initialise the SmartSim Experiment and the SmartSim Orchestrator.

Experiments are the Python user interface for SmartSim.
Experiment is a factory class that creates stages of a workflow
and manages their execution.
The instances created by an Experiment represent executable code
that is either user-specified, like the ``Model`` instance created
by ``Experiment.create_model``, or pre-configured, like the ``Orchestrator``
instance created by ``Experiment.create_database``.
Experiment methods that accept a variable list of arguments, such as
``Experiment.start`` or ``Experiment.stop``, accept any number of the
instances created by the Experiment.
In general, the Experiment class is designed to be initialized once
and utilized throughout runtime.

The Orchestrator is an in-memory database that can be launched
alongside entities in SmartSim. Data can be transferred between
entities by using one of the Python, C, C++ or Fortran clients
within an entity.
"""

import socket
import subprocess

from smartsim import Experiment
from smartsim.database import Orchestrator
from smartsim.log import get_logger

logger = get_logger(__name__)

def get_host():
    """
    Get the host the script is executed on from the env variable
    """
    return socket.gethostname()


def get_slurm_hosts():
    """
    Get the host list from the SLURM_JOB_NODELIST environment variable
    """
    hostslist_str = subprocess.check_output(
        "scontrol show hostnames", shell=True, text=True
    )
    return list(set(hostslist_str.split("\n")[:-1]))  # returns unique name of hosts


def write_hosts(hosts, n_slots, hostfile='hostfile'):
    # TODO: check if this works, of if i should do "localhost slots=.." as before
    with open(hostfile, 'w') as f:
        for host in hosts:
            f.write(f"{host} slots={n_slots}\n") # max_slots=4\n")
    with open(hostfile, 'r') as f:
        logger.debug(f"Edited hostfile '{hostfile}' as: \n{f.read()}")


def get_slurm_walltime():
    """
    Get the walltime of the current SLURM job
    """
    cmd = 'squeue -h -j $SLURM_JOBID -o "%L"'
    return subprocess.check_output(cmd, shell=True, text=True)[:-2] # it ends with '\n'


def init_smartsim(
    port = 6790,
    num_dbs = 1,
    network_interface = "lo",
    launcher = "local",
    run_command = "mpirun",
):
    """
    Initializes the SmartSim architecture by starting the Orchestrator, launching the Experiment, and get the list of hosts.
    """
    logger.info("Starting SmartSim...")

    launchers = []

    # Launch in a local environment
    if launcher == "local":
        db_is_clustered = False
        hosts = [get_host()]
        logger.info(f"Launching locally on: {hosts}")

    # Launch in a cluster environment
    elif launcher in launchers:
        assert network_interface != "lo", "Launching in a cluster but network interface is set as 'lo'."
        hosts = get_slurm_hosts()
        num_hosts = len(hosts)
        logger.info(f"Launching in cluster. Identified available nodes: {hosts}")
        # Is database clustered, i.e. hosted on different nodes?
        db_is_clustered = num_dbs > 1
        try:
            # Get slurm settings
            walltime = get_slurm_walltime()
            # Maximum of 1 DB per node allowed for Slurm Orchestrator
            if num_hosts < num_dbs:
                logger.warning(
                    f"You selected {num_dbs} databases and {num_hosts} nodes, but maximum is 1 database per node.\
                    \nSetting number of databases to {num_hosts}"
                )
                num_dbs = num_hosts
            # Clustered DB with Slurm orchestrator requires at least 3 nodes for reasons
            if db_is_clustered:
                if num_dbs < 3:
                    logger.warning(
                        f"Only {num_dbs} databases requested, but clustered orchestrator requires 3 or more databases.\
                        \nNon-clustered orchestrator is launched instead!"
                    )
                    db_is_clustered = False
                else:
                    logger.info(f"Using a clustered database with {num_dbs} instances.")
            else:
                logger.info("Using an unclustered database on root node.")
        except Exception as exc:
            # If there are no environment variables for a batchjob, then use the local launcher
            raise ValueError("Didn't find SLURM batch environment.") from exc

    # Environment not found.
    else:
        raise ValueError(f"Launcher type '{launcher}' not implemented.")

    # Define smartsim launcher
    smartsim_launcher = "local" if launcher == "local" else "slurm"

    # Generate Sod2D experiment
    """ Initialize RHEA experiment instance
            Experiment is a factory class that creates stages of a workflow
            and manages their execution.
            __init__(self, name, exp_path=None, launcher='local')
            :param name: name for the ``Experiment``
            :type name: str
            :param exp_path: path to location of ``Experiment`` directory if generated
            :type exp_path: str, optional
            :param launcher: type of launcher being used, options are "slurm", "pbs", "cobalt", "lsf", or "local". 
                If set to "auto", an attempt will be made to find an available launcher on the system. Defaults to "local"
            :type launcher: str, optional  """
    exp = Experiment("rhea_exp", launcher=smartsim_launcher)

    """ Initialize an Orquestrator reference for local launch
            The Orchestrator is an in-memory database that can be launched alongside entities in SmartSim. Data can be transferred between
            entities by using one of the Python, C, C++ or Fortran clients within an entity.
            __init__(self, port=6379, interface='lo', launcher='local', run_command='auto', db_nodes=1, batch=False, hosts=None, 
                     account=None, time=None, alloc=None, single_cmd=False, **kwargs)
            :param port: TCP/IP port, defaults to 6379
            :type port: int, optional
            :param interface: network interface, defaults to "lo"
            :type interface: str, optional

            Extra configurations for RedisAI
            See https://oss.redislabs.com/redisai/configuration/
            :param threads_per_queue: threads per GPU device
            :type threads_per_queue: int, optional
            :param inter_op_threads: threads accross CPU operations
            :type inter_op_threads: int, optional
            :param intra_op_threads: threads per CPU operation
            :type intra_op_threads: int, optional  """
    if launcher=="local":
        db = Orchestrator(port=port, interface="lo")
    else:
        db = Orchestrator(
                launcher=smartsim_launcher,
                port=port,
                db_nodes=num_dbs,               # SlurmOrchestrator supports multiple databases per node
                batch=db_is_clustered,          # false if it is launched in an interactive batch job
                time=walltime,                  # this is necessary, otherwise the orchestrator wont run properly
                interface=network_interface,    # 
                hosts=hosts,                    # specify hostnames of nodes to launch on (without ip-addresses)
                run_command=run_command,        # ie. mpirun, srun, etc
                db_per_host=1,                  # number of database shards per system host (MPMD), defaults to 1
                single_cmd=True,                # run all shards with one (MPMD) command, defaults to True
        )

    # remove db files from previous run if necessary
    logger.info("Removing stale files from old database...")
    db.remove_stale_files()

    # create an output directory for the database log files
    logger.info("Creating output directory for database log files...")
    exp.generate(db, overwrite=True)

    # startup Orchestrator
    logger.info("Starting database...")
    exp.start(db)

    logger.info("If the SmartRedis database isn't stopping properly you can use this command to stop it from the command line:")
    for db_host in db.hosts:
        logger.info(f"$(smart dbcli) -h {db_host} -p {port} shutdown")

    logger.info(f"DB address: {db.get_address()[0]}")
    
    logger.info("Created 'rhea_exp' directory containing files: 'dataset/orchestrator_0.err', 'dataset/orchestrator_0.out', 'dataset/smartsim_db.dat'")
    logger.info("New file 'rhea_exp/dataset/smartsim_db.dat' example content: \n{'db': <smartsim.database.orchestrator.Orchestrator object at 0x7a8ae990b880>, 'db_jobs': {'orchestrator_0': <smartsim._core.control.job.Job object at 0x7a8ae8d26520>}, 'steps': [StepMap(step_id=None, task_id='498093', managed=False)]}")
    
    # we run the model on a single host, and have
    return exp, hosts, db, db_is_clustered
