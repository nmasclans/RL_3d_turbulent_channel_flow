## Managing running processes

You might need to kill running processes in case of unsuccessful process termination, e.g. redis and mpirun processes. 

### Finding the Redis Process

To locate the Redis process running on your system, you can use the following command to search for the Redis server:

```sh
ps aux | grep redis-server
```

This command will list processes related to redis-server. Look for the line that contains redis-server in the output. The second column in this output represents the process ID (PID) of the Redis server.

### Finding the Mpirun Process

To locate the mpirun process running on your system, do:
```sh
ps aux | grep mpirun
```

### Killing the Process

Once you have identified the PID of the running process, you can terminate it using the kill command. Replace <PID> with the actual PID you found:

```sh
sudo kill <PID>
```

If the process does not terminate with the standard kill command, you can use the -9 option for a more forceful termination:

```sh
sudo kill -9 <PID>
```

## Check content smartredis database from file

To check the content of smartredis database from the database file `smartsim_db.dat`, do:
```bash
conda activate smartrhea-env; python3
```
```python3
import pickle
with open("smartsim_db.dat","rb") as file:
    data = pickle.load(file)
print(data)
```