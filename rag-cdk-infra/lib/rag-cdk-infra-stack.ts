import * as cdk from "aws-cdk-lib";
import { Construct } from "constructs";
import * as fs from 'fs';
import * as path from 'path';
import { AttributeType, BillingMode, Table } from "aws-cdk-lib/aws-dynamodb";
import {
  DockerImageFunction,
  DockerImageCode,
  FunctionUrlAuthType,
  Architecture,
} from "aws-cdk-lib/aws-lambda";
import { ManagedPolicy } from "aws-cdk-lib/aws-iam";
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as iam from 'aws-cdk-lib/aws-iam';

export class RagCdkInfraStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);


    // Get the environment variables from the Python config file.
    const pineconeApiKey = process.env.PINECONE_API_KEY;
    if (!pineconeApiKey) {
      throw new Error("PINECONE_API_KEY environment variable is not set");
    }

    // Create a DynamoDB table to store the query data and results.
    // const ragQueryTable = new Table(this, "QueriesTable", {
    //   partitionKey: { name: "query_id", type: AttributeType.STRING },
    //   billingMode: BillingMode.PAY_PER_REQUEST,
    // });
    
    // Using the table already created in the previous step.
    const ragQueryTable = Table.fromTableName(this, "ExistingQueriesTable", "RagCdkInfraStack-QueriesTable7395E8FA-17BGT2YQ1QX1F");

    // Create a new DynamoDB table for processed files
    const processedFilesTable = new Table(this, "ProcessedFilesTable", {
      tableName: "lewas-chatbot-processed-files",
      partitionKey: { name: "filename", type: AttributeType.STRING },
      billingMode: BillingMode.PAY_PER_REQUEST,
    });

    // Function to handle the API requests. Uses same base image, but different handler.
    const apiImageCode = DockerImageCode.fromImageAsset("../image", {
      cmd: ["main.handler"]
    });
    const apiFunction = new DockerImageFunction(this, "ApiFunc", {
      code: apiImageCode,
      memorySize: 256,
      timeout: cdk.Duration.seconds(30),
      architecture: Architecture.ARM_64,
      environment: {
        ...this.getConfigFromPython(),
        TABLE_NAME: ragQueryTable.tableName,
        PINECONE_API_KEY: pineconeApiKey,
      },
    });

    // Grant Bedrock permissions to the API function
    // const bedrockPolicy = new iam.PolicyStatement({
    //   effect: iam.Effect.ALLOW,
    //   actions: [
    //     'bedrock:InvokeModel',
    //     'bedrock:ListFoundationModels',
    //     'bedrock:GetFoundationModel'
    //   ],
    //   resources: ['*']  // You might want to restrict this to specific model ARNs
    // });

    // apiFunction.addToRolePolicy(bedrockPolicy);
    apiFunction.role?.addManagedPolicy(
      ManagedPolicy.fromAwsManagedPolicyName("AmazonBedrockFullAccess")
    );


    // Public URL for the API function.
    const functionUrl = apiFunction.addFunctionUrl({
      authType: FunctionUrlAuthType.NONE,
    });

    // Pemission for the API fucntion to access the S3 bucket.
    // Reference the existing S3 bucket
    const existingBucket = s3.Bucket.fromBucketName(this, 'ExistingBucket', 'lewas-chatbot');

    // Grant read/write permissions to the API function
    existingBucket.grantReadWrite(apiFunction);


    // Add S3 environment variables to both functions
    apiFunction.addEnvironment('S3_BUCKET', 'lewas-chatbot');

    apiFunction.role?.addManagedPolicy(
      ManagedPolicy.fromAwsManagedPolicyName("AmazonS3FullAccess")
    );

    const lanceDbPolicy = new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        's3:GetBucketLocation',
        's3:ListBucket',
        's3:ListBucketMultipartUploads',
        's3:ListMultipartUploadParts',
        's3:AbortMultipartUpload',
        's3:CreateMultipartUpload',
        's3:PutObject',
        's3:GetObject',
        's3:DeleteObject'
      ],
      resources: [
        `arn:aws:s3:::${existingBucket.bucketName}`,
        `arn:aws:s3:::${existingBucket.bucketName}/*`
      ]
    });
    
    apiFunction.addToRolePolicy(lanceDbPolicy);

    // Grant permissions for all resources to work together.
    ragQueryTable.grantReadWriteData(apiFunction);
    processedFilesTable.grantReadWriteData(apiFunction);

    // Output the URL for the API function.
    new cdk.CfnOutput(this, "FunctionUrl", {
      value: functionUrl.url,
    });
  }

  private getConfigFromPython(): { [key: string]: string } {
    const configPath = path.join(__dirname, '..', '..', 'image','src', 'config.py');
    const configContent = fs.readFileSync(configPath, 'utf8');
  
    const envVars: { [key: string]: string } = {};
    const lines = configContent.split('\n');
    
    const reservedVars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_REGION'];
    
    for (const line of lines) {
      const match = line.match(/(\w+)\s*=\s*os\.getenv\("(\w+)"(?:,\s*"([^"]*)")?\)/);
      if (match) {
        const [, varName, envName, defaultValue] = match;
        if (!reservedVars.includes(envName)) {
          envVars[envName] = process.env[envName] || defaultValue || '';
        }
      }
    }
  
    return envVars;
  }

}
