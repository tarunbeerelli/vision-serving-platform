output "cluster_name" {
  description = "GKE cluster name"
  value       = module.gke.cluster_name
}

output "cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = module.gke.cluster_endpoint
  sensitive   = true
}

output "model_bucket_name" {
  description = "GCS bucket for Triton model repository"
  value       = google_storage_bucket.model_repo.name
}

output "node_service_account" {
  description = "GKE node service account email"
  value       = module.iam.node_service_account_email
}
