provider "google" {
  project = var.project_id
  region  = var.region
}

# ── VPC ──────────────────────────────────────────────────────────────────────
module "vpc" {
  source       = "./modules/vpc"
  project_id   = var.project_id
  region       = var.region
  cluster_name = var.cluster_name
}

# ── GKE cluster ──────────────────────────────────────────────────────────────
module "gke" {
  source               = "./modules/gke"
  project_id           = var.project_id
  region               = var.region
  zone                 = var.zone
  cluster_name         = var.cluster_name
  network              = module.vpc.network_name
  subnetwork           = module.vpc.subnet_name
  node_service_account = module.iam.node_service_account_email
}

resource "google_service_account_iam_member" "triton_workload_identity" {
  service_account_id = module.iam.triton_service_account_id
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[triton/triton-server]"

  depends_on = [module.gke] # wait for cluster so the identity pool exists
}

# ── IAM + Workload Identity ───────────────────────────────────────────────────
module "iam" {
  source            = "./modules/iam"
  project_id        = var.project_id
  cluster_name      = var.cluster_name
  model_bucket_name = var.model_bucket_name
  github_repo       = var.github_repo
}

# ── GCS model bucket ─────────────────────────────────────────────────────────
# Triton reads models from here via init container
resource "google_storage_bucket" "model_repo" {
  name                        = var.model_bucket_name
  location                    = var.region
  force_destroy               = false
  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }
}

# Grant Triton's k8s service account read access to the model bucket
resource "google_storage_bucket_iam_member" "triton_model_reader" {
  bucket = google_storage_bucket.model_repo.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${module.iam.triton_workload_identity_member}"
}
