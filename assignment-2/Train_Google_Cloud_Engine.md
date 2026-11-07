# Training a Neural Network with Google Cloud

There are many ways to train models with Google Cloud, e.g., using a virtual machine in the Compute Engine, using the 
cloud ML Engine, etc. Here, we recommend that you use a Virtual Machine (VM) as that is the closest
setup to working with a local machine.

> All the students that are enrolled in CPSC-4590/5590 will receive a $50 credit for Google Cloud
to use for Assignment 2 or their final project. The credit will be associated
to the student's Yale email. Students should use the resources judiciously. 

> We do not recommend that you try to work on the ROS components of Assignment 2 (or your projects) using a Google Cloud virtual machine because it will likely end up being very expensive. Instead, we suggest you leverage the cloud for training ML models or compute-heavy work.

## How to claim your BIM credits?

1. Redeem the coupon while signed in to Google Cloud with your yale.edu account. The coupon redemption process will create a billing account ("Billing Account for Education") with the credits on it. Within this billing account, if you go to the "Credits" page, you should see the BIM credits listed as "CPSC 4590/5590 Building Interactive Machines".

2. Create a project: in the Google Cloud console, click New Project, and under Location choose `yale.edu > CPSC-4590`. You may need to search for this folder as "CPSC-4590" in the search bar for project locations.

3. Link your credits: in the new project, go to Billing and link the billing account from step 1.


## WARNING. Make Sure to Stop Your VM Instances!

A VM instance costs money for as long as it exists, not only while you are actively using it. The charges come from several resources:

- **Compute:** the CPUs, memory, and GPU attached to the VM. This is usually the largest cost, and you are charged for it the whole time the VM is running, even if it's sitting idle.
- **Storage:** the boot disk (and any other disks) attached to the VM. You pay for disk space as long as the disk exists, even when the VM is stopped.
- **Network:** external IP addresses and data sent out of Google Cloud (e.g., downloading results to your computer).

If you leave a VM running, charges keep accumulating and you can quickly run out of credits. To avoid this:

- **If you will keep using the VM over a period of time but only work on it now and then** (e.g., a few hours a day while working on Assignment 2), **stop it every time you finish working** and start it again when you come back. You can do this with the stop and start buttons at the top of the [compute instances page](https://console.cloud.google.com/compute/instances) showing your instances. While the VM is stopped, you are not charged for compute (including the GPU), so costs drop significantly. You will still pay a small amount for the disk, and your files on it will still be there when you restart the VM.

- **If you don't need the VM anymore**, **delete it**, along with all the resources it used (e.g., its disks, any snapshots or images you created from it, and any reserved static IP addresses). Deleting the VM is the only way to bring its charges down to zero. Before deleting, make sure you have saved anything you need (e.g., push your code to GitHub and download your trained models).

<img src="docs/no-more-credits.jpeg" width="300"/><br/>
(image created by Gemini 3.6 Thinking)

## Getting Started with the Compute Engine of Google Cloud

You can create Virtual Machine (VM) instances via the [Compute Engine web interface](https://console.cloud.google.com/compute/overview) in Google Cloud; however, there are some important considerations to keep in mind for Yale accounts:

- New projects in your Google Cloud Yale account get no "default" network, and this will prevent you from choosing a network interface for a new VM via the web interface. This means that you will need to create a network prior to creating the VM instance, and this network must have a subnet that matches the region of your VM. 

- Every subnet has to have VPC flow logs on at 100% sampling, so a network created with the default settings would get rejected.

- VMs can't have public IP addresses, so set External IPv4 to "None". You can still SSH in through [Google's IAP tunnel](https://docs.cloud.google.com/iap/docs).

- Because the VM has no public IP, it can't reach the internet on its own. A [Cloud Network Address Translation (NAT)](https://docs.cloud.google.com/nat/docs/overview) gateway lets it make outgoing connections, so git, pip and apt still work.  

Another important detail for Google Cloud is that the cheaper GPUs (like NVidia T4 GPUs with 16GB) are scarce. The L4 GPUs (with 24 GB) are more available, but not always. Thus, creating the network and the VM by hand in the Google Cloud web interface can be a very tedious and time-consuming process:  you choose a zone to try, create the network, create a VM's within that zone with the GPU you want, and then fail because the accelerator is currently unavailable in that zone, so you repeat the process... For this reason, we recommend instead using the Google Cloud CLI to create networks and VMs. We provide you scripts to facilitate this process, as explained below.

As of writing of this assignment, we have had luck creating VMs in the `us-central1-b` zone with `g2-standard-4` (4 vCPUs, 16 GB Memory) machines that have a `NVIDIA L4 GPU` (with 24 GB of GPU memory). This machine costs approximately $0.71/hour, so your credit covers roughly 70 hours of running time. Leaving this VM on for a day costs about $17. 

You can see a description of the GPU machine types that  Google Cloud offers [here](https://docs.cloud.google.com/compute/docs/gpus?hl=en#:~:text=Compute%20Engine%20offers%20different%20machine%20types%20to%20support%20your%20various%20workloads.).
 
### Installing the Google Cloud CLI

Download the Google Cloud CLI following the instructions in [this page](https://docs.cloud.google.com/sdk/docs/install-sdk?_gl=1*1hvo9vc*_up*MQ..*_gs*MQ..&gclid=CjwKCAjwuobXBhBXEiwAYUaIlbW7_rfn3_8fVzyIGwwxEA7FKX3mzYGAR2qLfgOYNsxchR7xIr3aYRoCZCAQAvD_BwE&gclsrc=aw.ds&hl=en). This should give you the command line tool `gcloud`. Note that if you install from an archive (following [these instructions](https://docs.cloud.google.com/sdk/docs/downloads-versioned-archives)), then you should not need admin permissions on the machine that you are using to access and manage Google Cloud.

### Requesting GPU quotas

1. First, check whether you have requested quota for a GPU like the `NVIDIA L4 GPU`:

```bash
$ gcloud compute regions describe us-central1 --project=PROJECT_ID --flatten=quotas --format="value(quotas.metric,quotas.limit,quotas.usage)" | grep NVIDIA_L4_GPUS
```

The columns are the metric name, the limit and how much is in use. A limit of 0.0 means you need to request quota.

2. Enable the Cloud Quotas APU (one per project):

```bash
$ gcloud services enable cloudquotas.googleapis.com --project=PROJECT_ID
```

3. Submit the request:

```bash
$ gcloud beta quotas preferences create \
  --service=compute.googleapis.com \
  --quota-id=GPUS-ALL-REGIONS-per-project \
  --preferred-value=1 \
  --project=PROJECT_ID \
  --justification="Training a neural network for a Yale course assignment (CPSC 4590/5590)." \
  --email=YOUR_NETID_EMAIL@yale.edu
```

If you get a message that says that the quota already exists, you can list the project's existing quota requests with:

```bash
$ gcloud beta quotas preferences list --project=PROJECT_ID \
   --format="table(name.basename(),quotaId,quotaConfig.preferredValue,quotaConfig.grantedValue,reconciling)"
```

### Creating a network and VM

You can run specific commands to create a network and a VM with `gcloud`. For example, to create a network in `us-east4`, you could run in your terminal:

```bash
$ gcloud config set project YOUR_PROJECT_ID
$ gcloud compute networks create vpc-network --subnet-mode=custom
$ gcloud compute networks subnets create subnet-a --network=vpc-network --region=us-east4 --range=192.168.1.0/24 --enable-flow-logs --logging-flow-sampling=1.0 --enable-private-ip-google-access
$ gcloud compute firewall-rules create allow-ssh-iap --network=vpc-network --allow=tcp:22 --source-ranges=35.235.240.0/20
$ gcloud compute routers create cloud-router --network=vpc-network --region=us-east4
$ gcloud compute routers nats create cloud-nat --router=cloud-router --region=us-east4 --auto-allocate-nat-external-ips --nat-all-subnet-ip-ranges
```

Then, to create a VM with Ubuntu 24.04, you could run:

```bash
$ gcloud compute instances create my-gpu-vm --zone=us-east4-a --machine-type=g2-standard-4 \
      --subnet=subnet-a --no-address --maintenance-policy=TERMINATE \
      --image-family=ubuntu-2404-lts-amd64 --image-project=ubuntu-os-cloud \
      --metadata=VmDnsSetting=ZonalOnly \
      --boot-disk-size=50GB --quiet
```

Because it is hard to predict which zone will have such machine available, we created example scripts in the [gcloud](gcloud) folder to help with this process:

- [create_vm.sh](gcloud/create_vm.sh): Creates a Google Cloud VM with an Nvidia L4 GPU (default), an Nvidia T4 GPU, or no GPU for Assignment 2 that is named `my-gpu-vm`. It searches through different zones to try to create the VM, and in the process, creates the network as well. Notably, this script pre-installs the NVidia drivers and the python requirements in [install_python_deps.sh](install_python_deps.sh) so that when you log into the machine, you can start working on training and testing neural networks.
- [stop_vm.sh](gcloud/stop_vm.sh): Stops a VM with name `my-gpu-vm`. If you provide the `--delete` argument to the script, it also deletes all the resources needed to operate that VM (the network, hard drive, etc.) and the VM itself.

Thus, to get started with creating a virtual machine, you can simply run:

```bash
$ cd gcloud
$ ./create_vm.sh <PROJECT-NAME>        # L4 GPU on a g2-standard-4 machine
$ ./create_vm.sh <PROJECT-NAME> t4     # T4 GPU on an n1-standard-4 machine
$ ./create_vm.sh <PROJECT-NAME> cpu    # no GPU, n1-standard-4 machine
```

If the script can't find an available L4 GPU in any zone, try again with `t4` (or vice versa). As a last resort, `cpu` creates a VM without a GPU. It is almost always available, but training runs much more slowly on it.

To connect to your VM via SSH once it's running:

```bash
$ gcloud compute ssh my-gpu-vm --zone=<VM-ZONE> --project=<PROJECT-NAME> --tunnel-through-iap
```

Then, when you want to stop working with the VM:

```bash
$ cd gcloud
$ ./stop_vm.sh <PROJECT-NAME> [--delete]
```

To start the VM again, if it was not deleted, you can run:

```bash
$ gcloud compute instances start my-gpu-vm --zone=<VM-ZONE>
```

Note that stopping a GPU VM gives up its GPU, and starting it again can fail if the zone where it was created stops having capacity for it. You can then choose to wait to start the VM again later, or delete it and try to create a new one.

### TensorBoard 
It is also possible to use port forwarding (as discussed in [here](https://www.apoehlmann.com/blog/connecting-tensorboard-your-google-cloud-vm-port-forwarding/#:~:text=PC-,So,6006%2E,-For)) to 
run TensorBoard on a Google Cloud VM and visualize your results on a local browser.
To succeed running the steps below, you will need to install the [Google Cloud (gcloud) sdk](https://cloud.google.com/sdk/gcloud/).

1. On a terminal in your host machine, authenticate and log into your VM:
    ```bash
    $ gcloud auth login
    $ gcloud compute ssh <vm-name> --zone <zone>
    ```
    > Tip: You can avoid passing `--zone` every time by running `gcloud config set compute/zone <zone>`.
    If you have multiple Google Cloud projects, also pass `--project <project-id>`.

2. On the VM (inside the SSH session from step 1), start TensorBoard on port 6006:
    ```bash
    $ tensorboard --logdir <path_to_logs> --port 6006
    ```

3. Open another terminal in your host machine and forward your local port 6006 to port 6006 on the VM:
    ```bash
    $ gcloud compute ssh <vm-name> --zone <zone> -- -NfL 6006:localhost:6006
    ```
    The `-L` flag sets up local port forwarding through SSH, `-N` tells SSH not to run a remote command,
    and `-f` sends the tunnel to the background. Because the connection goes through SSH, you do not
    need to open port 6006 in the VM's firewall.

4. Finally, in your browser navigate to: 
    ```
    http://localhost:6006/
    ```

5. When you are done working, stop TensorBoard on the VM by pressing `Ctrl+C` in the terminal where it is running.
Then, close the SSH tunnel on your host machine. Because the tunnel was sent to the background (`-f`), closing
the terminal does not stop it, so you need to kill the process explicitly:
    ```bash
    $ pkill -f "6006:localhost:6006"
    ```
    You can verify that the tunnel is gone by checking that no process is listening on port 6006 anymore 
    (the command below should print nothing):
    ```bash
    $ lsof -i :6006
    ```
    > Tip: Remember to also stop your VM from the Google Cloud console (or with `gcloud compute instances stop <vm-name> --zone <zone>`)
    when you are not using it, so you don't use up your credits.
    


