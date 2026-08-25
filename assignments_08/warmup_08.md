
##----------------------------------------------Part 1: Warmup — Cloud Concepts-------------------------------------------------------------

##-------------------------------------------------Cloud Concepts Question 1--------------------------------------------------

""" 
Question: What is the core economic model of cloud computing, and how does it differ from owning your own servers?

Answer: The core economic model of cloud computing is based on a pay as you go model, where the user only pays what the user uses.
        owning your own servers needs more investment and maintainance than just paying for what you use in a cloud environment.


"""

##-------------------------------------------------Cloud Concepts Question 2--------------------------------------------------

"""
Question: What is the difference between vertical scaling and horizontal scaling? Give a concrete example of when you might choose each.

Answer: Vertical scaling  is when we add more resources like add more ram or gpu or storeage to a single server.
        Horizontal scaling is when we add more servers insstad of adding more resources to a single server.

        we choose vertical scaling when a single database server is running slow or out of memory then we can choose virtual scaling. 
        we choose horizontal scaling when we have a web application that needs to handle 100,000 after a viral product launch
        where it normally handles 1000 users per day. 
        

scenario: A web app that normally handles 1,000 users per day suddenly needs to handle 100,000 after a viral product launch.

Answer: For this scenario,i would choose  horizontal scaling so the traffic can be distributed across multiple servers.


"""

##-------------------------------------------------Cloud Concepts Question 3--------------------------------------------------

""" 

Gmail : Gmail is SaaS (Software as a Service) because it is a software application that we use over the internet and
        does not require any installation or maintenance by the user.

Azure Virtual Machines : Azure Virtual Machines is IaaS (Infrastructure as a Service) because it provides virtualized 
                        computing resources over the internet, allowing users to run their own applications and operating systems 
                        without managing the underlying hardware.

AWS S3 (Simple Storage Service) : AWS S3 is BaaS (Backend as a Service) because it provides cloud storage and data management services,
                                 allowing developers to give their app a backend without managing the underlying infrastructure.

GitHub Codespaces : GitHub Codespaces is PaaS (Platform as a Service) because it provides a cloud-based development environment
                    that allows developers to write, build, and test code without managing the underlying infrastructure.

Snowflake : Snowflakes is SaaS (Software as a Service) because it is a cloud based and it gives services like pre-wire the pieces for you, 
           optimizing specifically for data and analytics workloads.


Supabase : Supabase is open-source  BaaS (Backend as a Service) because it provides a backend infrastructure for developers 
           to build and deploy applications without managing the underlying servers or databases.

           
IaaS : Iaas stands for Infrastructure as a Service, Where the cloud service provider gives the user access to a virtualized computing infrastructure.
        where the user pay what they use and don't need to manage the underlying hardware.
        Eg: Google Cloud Compute Engine.


PaaS : PaaS stands for Platform as a Service, where the cloud service provider gives the user a platform where the 
       can write, run, test and deploy their applications without managing the underlying infrastructure.         
        Eg: GitHub Codespaces.


SaaS : SaaS stands for Software as a Service, where the cloud service provider gives the user access to complete software 
       applications over the internet.
        Eg: Gmail.

"""


##-------------------------------------------------Cloud Concepts Question 4--------------------------------------------------

""" 

Managed data platform like Databricks or Snowflake are a managed data platfrorm that sits on top of the cloud infrastructure 
and provides a platform for data processing, analytics, and machine learning without the need to manage the underlying infrastructure.

The gain here is it's faster to get started and it manages cloud resources on my behalf.
The tradeoff is its less flexible and the cost can be higher.

"""

##-------------------------------------------------Cloud Concepts Question 5--------------------------------------------------

"""  

First situation where the dataset fits comfortably on a single machine and do not have massive compute demands.

Second where the user is setting up an initial prototype.

"""


##-----------------------------------------------Part 2: Warmup — Cloud Landscape---------------------------------------------------


##------------------------------------------------Cloud Landscape Question 1-------------------------------------------------------

"""  
The 3 hyperscalers are:

1. Amazon Web Services (AWS), Its primary strengths are EC2 (compute), S3 (object storage), RDS (managed databases),
   SageMaker (ML platform) and a large organization or a startup can use it.

2. Microsoft Azure, Its primary strengths are enterprise and government settings, largely because of its deep integration with Windows.

3. Google Cloud Platform (GCP), Its primary strengths are that it is strongest in data and machine learning, organizations that
   are data driven or AI focused can use it.


"""

##------------------------------------------------Cloud Landscape Question 2-------------------------------------------------------

"""  

1.  Access: Azure used to take long time to  joining a tenant, waiting for an invitation, configuring authentication wehere Supabse can do it 
    in a few minutes.

2. Pedagogical fit: Azure service used to stores data as opaque files. Supabase stores data as rows and columns in a relational database.
   A relational database is more transferable querying, filtering, and reasoning about structured data.

3. Pipeline coherence: Supabase makes the ETL pipeline easier to understand because the raw and enriched data can be stored in two tables 
   that are easy to look at and debug.

Reflection : It suggests that just choosing a service provider based on popularity isn't always the best choice. We should look at how easy 
             it is to use, the cost, how well it fits the project.


"""


##------------------------------------------------Cloud Landscape Question 3-------------------------------------------------------

""" 
1. Object storage offerred by AWS S3, GCP Cloud Storage.

2. ML platform offered by AWS SageMaker, GCP Vertex AI.

3. Serverless compute offered by AWS Lambda, GCP Cloud Functions.

4. LLM API offered by AWS Bedrock, GCP Vertex AI

"""

##------------------------------------------------Cloud Landscape Question 4-------------------------------------------------------

"""  

I could build a customer feedback dashboard that gathers feedback from a web app, stores the data using AWS, 
looks for trends, and uses a GCP LLM to summarize what customers are saying.


"""




