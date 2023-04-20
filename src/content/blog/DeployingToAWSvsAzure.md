---
title: "Beginner's Guide for Deploying a Sample Application to AWS and Azure"
description: "A step-by-step guide with scripts for beginners to deploy a sample application to AWS and Azure cloud platforms..."
pubDate: "Apr 20 2023"
heroImage: "/post_img.webp"
---
Deploying a sample application to AWS or Azure can seem daunting, but with the right guidance, it can be done easily. In this guide, we'll walk you through the steps to deploy a sample application to both AWS and Azure, including sample scripts to make the process even easier.

Sample Application
We'll be using a sample application called "Hello World" for this tutorial. It's a simple web application that displays a "Hello, World!" message when accessed through a web browser.

**AWS Deployment**  

Step 1: Create an AWS Account
To deploy the sample application to AWS, you'll need to first create an AWS account. If you don't already have one, you can sign up for a free account at https://aws.amazon.com/.

Step 2: Create an EC2 Instance
Next, you'll need to create an EC2 instance to host the sample application. Follow these steps:  
1. Log in to your AWS account and go to the EC2 dashboard.  
2. Click on "Launch Instance" and select an Amazon Machine Image (AMI) that meets your needs. For this tutorial, we'll use the "Amazon Linux 2 AMI (HVM), SSD Volume Type".  
3. Choose an instance type that meets your needs. For this tutorial, we'll use the t2.micro instance type.
4. Configure the instance details, including the VPC and subnet.
5. Add storage and configure any additional settings as needed.
6. Review and launch the instance.

Step 3: Install and Configure Web Server  
Once the instance is up and running, you'll need to install and configure a web server to host the sample application. Follow these steps:  
1. SSH into the EC2 instance using the command line or a tool like PuTTY.
2. Install the Apache web server using the following command:  
sudo yum install httpd  
3. Start the Apache web server using the following command:
sudo service httpd start  
4. Copy the sample application files to the web server's document root. For this tutorial, we'll use the default document root at /var/www/html/.

Step 4: Access the Sample Application
Finally, you can access the sample application by navigating to the public IP address of your EC2 instance in a web browser. You should see the "Hello, World!" message displayed.  

**Azure Deployment**  

Step 1: Create an Azure Account  
To deploy the sample application to Azure, you'll need to first create an Azure account. If you don't already have one, you can sign up for a free account at https://azure.microsoft.com/.

Step 2: Create an Azure VM  
Next, you'll need to create a virtual machine (VM) to host the sample application. Follow these steps:  
1. Log in to your Azure account and go to the Azure portal.
2. Click on "Create a resource" and search for "Ubuntu Server".
3. Select "Ubuntu Server" from the list of available resources and click "Create".
4. Configure the VM details, including the name, resource group, and authentication settings.
5. Choose an appropriate size for the VM. For this tutorial, we'll use the Standard B1s size.
6. Configure additional settings as needed, such as networking and storage.
7. Review and create the VM.

Step 3: Install and Configure Web Server  
Once the VM is up and running, you'll need to install and configure a web server to host the sample application. Follow these steps:  
1. SSH into the Azure VM using the command line or a tool like PuTTY.  
2. Install the Apache web server using the following command:  
sudo apt-get update  
sudo apt-get install apache2
3. Start the Apache web server using the following command:  
sudo systemctl start apache2  
4. Copy the sample application files to the web server's document root. For this tutorial, we'll use the default document root at /var/www/html/.

Step 4: Access the Sample Application
Finally, you can access the sample application by navigating to the public IP address of your Azure VM in a web browser. You should see the "Hello, World!" message displayed.

**Scripts**  
To make the deployment process even easier, here's sample scripts for both AWS and Azure.

**AWS Script:**  
```bash
#!/bin/bash

# Install Apache web server  
sudo yum update -y
sudo yum install httpd -y
sudo systemctl start httpd.service
sudo systemctl enable httpd.service

# Copy sample application files to document root  
sudo rm -rf /var/www/html/*
sudo wget https://raw.githubusercontent.com/aws-samples/hello-aws-world/main/index.html -O /var/www/html/index.html
sudo wget https://raw.githubusercontent.com/aws-samples/hello-aws-world/main/hello.css -O /var/www/html/hello.css
```

**Azure Script:**  
```bash
#!/bin/bash

# Install Apache web server  
sudo apt-get update
sudo apt-get install apache2 -y
sudo systemctl start apache2
sudo systemctl enable apache2

# Copy sample application files to document root  
sudo rm -rf /var/www/html/*
sudo wget https://raw.githubusercontent.com/aws-samples/hello-aws-world/main/index.html -O /var/www/html/index.html
sudo wget https://raw.githubusercontent.com/aws-samples/hello-aws-world/main/hello.css -O /var/www/html/hello.css
```

Deploying a sample application to AWS or Azure can seem intimidating, but with the right steps and scripts, it's actually quite simple. By following this guide and using the provided scripts, you should be able to easily deploy the "Hello World" sample application to both AWS and Azure.