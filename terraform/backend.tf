# Remote state in GCS — never lose state, safe for team use.
# We create this bucket manually ONCE before running terraform init.
# It cannot be managed by Terraform itself (chicken-and-egg problem).
terraform {
  backend "gcs" {
    bucket = "vision-serving-tf-state-564530715752"
    prefix = "terraform/state"
  }

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  required_version = ">= 1.5.0"
}
