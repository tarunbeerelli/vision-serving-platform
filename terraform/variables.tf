variable "project_id" {
  description = "GCP project ID"
  type        = string
  default     = "vision-serving-platform"
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-east1"
}

variable "zone" {
  description = "GCP zone for GPU node pool — T4 availability varies by zone"
  type        = string
  default     = "us-east1-b"
}

variable "cluster_name" {
  description = "GKE cluster name"
  type        = string
  default     = "vision-serving"
}

variable "model_bucket_name" {
  description = "GCS bucket for Triton model repository"
  type        = string
  default     = "vision-serving-models-564530715752" # suffix with project number for global uniqueness
}

variable "github_repo" {
  description = "GitHub repo for OIDC — format: org/repo"
  type        = string
  default     = "tarunbeerelli/vision-serving-platform"
}
