---
title: "Algorithmic Trading with Python, Machine Learning, & AWS"
description: "The development of an algorithmic trading bot in Python using OANDA's API with different trading strategies and deploying it on AWS...."
pubDate: "Mar 23 2023"
heroImage: "/post_img.webp"
---

First of all, please create a free practice account from OANDA. After creating the practice account, you can execute trades with the real time data for free. 
To get the developer access token, please follow these steps:
1. Log in to your OANDA account at https://www.oanda.com/
2. Click on the "Manage API Access" button on the left-hand side of the dashboard.
3. Click on the "Create New Token" button.
4. Enter a name for your token and select the "Generate" button to create the token.
5. Once the token has been generated, copy the token and store it in a secure location.
6. You can now use the token to access the OANDA API. Be sure to keep the token secure and never share it with anyone.

Please find the GitHub repo at <a href="https://github.com/gaurav-aryal/AlgorithmicTradingWithOnada" target="_blank">AlgorithmicTradingWithOANDA</a>, download the code, make changes, and run the script in the AWS instance.

Now you can download the code. There are different strategies to chose from. I have implemented two strategies Simple Moving Averages and Mean Revision Strategy in NB_02_Many_Strategies.ipynb. And NB_03_Deep_Learning.ipynb implements a deep learning model based on past data. While the model predicts fairy accurately, it should be taken with a grain of salt since it is a very simple model. 
Trader.py implements Contrarian strategy which can be deployed in the cloud. You can also add fixed stop loss, trailing stop loss, time-based stop loss, volatility-based stop loss, and heding. You can add more functionality and make the strategy more complicated. You implement your own strategy on define_strategy method and add more features to the provided strategies. I have deployed Trader.py to the cloud in my case, but please feel free to modify and implement your own strategy.

After implementing and back-testing your strategy, you can deploy and your script in AWS. Here's a step-by-step guide on how to set up a free AWS account and run a Python script Trader.py on a Windows instance:

1. Go to aws.amazon.com and click on the "Create a Free Account" button. Follow the prompts to create an account with your email address, password, and payment information (even though it's free, you still need to provide a payment method).
2. Once you have an account, sign in to the AWS Management Console. You should see a dashboard with various services and resources available.
3. Click on the "EC2" service under "Compute" to create a new instance.
4. Click the "Launch Instance" button and select a Windows Server AMI (Amazon Machine Image).
5. Choose the free tier instance type, which is usually labeled "t2.micro".
6. In the "Configure Instance Details" section, choose the default settings unless you have specific requirements.
7. In the "Add Storage" section, choose the default settings unless you have specific requirements.
8. In the "Add Tags" section, add any relevant tags to help identify your instance.
9. In the "Configure Security Group" section, add a new rule to allow inbound traffic on port 3389 (Remote Desktop Protocol) from your IP address.
10. Click "Review and Launch" and then "Launch".
11. Create a new key pair (or use an existing one) to connect to your instance securely.
12. Once your instance is running, connect to it via Remote Desktop Protocol (RDP). You can do this from your local machine by opening the Remote Desktop app and entering your instance's public IP address.
13. Once connected, install Python and any required packages for your script. You can also upload your Python script to the instance via RDP or by using a file transfer service like WinSCP.
14. Finally, run your Python script Trader.py on the instance.

Congratulations! You've successfully set up a free AWS account, launched a Windows instance, and run the Python script that will automatically buy/sell instruments for you in New York Stock Exchange (NYSE) trading hours except on select holidays when the stock exchange is closed. Remember to monitor the trade either with your own Python scripts or at https://www.oanda.com/.