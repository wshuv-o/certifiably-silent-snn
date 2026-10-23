"""Launch CORE-SCALING-002 as a SageMaker training job (run in AWS CloudShell, region ap-southeast-2).
  pip install sagemaker && python sm_launch.py
Uses the approved ml.c6i.32xlarge training quota. Results: the job's model.tar.gz in S3 (manycore.log)."""
import boto3, sagemaker
from sagemaker.pytorch import PyTorch
REGION = "ap-southeast-2"
sess = sagemaker.Session(boto3.session.Session(region_name=REGION))
iam = boto3.client("iam")
# an execution role SageMaker can assume; service-linked roles (AWSServiceRoleFor...) cannot run jobs
roles = [r["Arn"] for r in iam.list_roles(MaxItems=1000)["Roles"]
         if "sagemaker" in r["RoleName"].lower() and not r["RoleName"].startswith("AWSServiceRole")
         and "sagemaker.amazonaws.com" in str(r.get("AssumeRolePolicyDocument", ""))]
if not roles:
    raise SystemExit("No SageMaker execution role found: create one in the SageMaker console (Admin configurations -> Role manager) and rerun.")
print("using role", roles[0])
est = PyTorch(entry_point="sm_entry.py", source_dir="bundle", role=roles[0], instance_type="ml.c6i.32xlarge",
              instance_count=1, framework_version="2.3", py_version="py311", max_run=3 * 3600,
              sagemaker_session=sess, base_job_name="speed006-probe",
              environment={"REPEATS": "2", "CORE_LIST": "4 8 16 32 64", "SCRIPT": "run_speed006_probe.sh", "LOGNAME_OUT": "probe6.log", "MODELS": "dense lk4 lk8 hdcls hscmd"})
est.fit(wait=True, logs="All")
print("model artifacts (manycore.log inside):", est.model_data)
