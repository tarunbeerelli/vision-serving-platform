output "node_service_account_email" {
  value = google_service_account.gke_node.email
}

output "triton_workload_identity_member" {
  value = google_service_account.triton.email
}

output "github_actions_service_account" {
  value = google_service_account.github_actions.email
}

output "workload_identity_provider" {
  value = google_iam_workload_identity_pool_provider.github.name
}

output "triton_service_account_id" {
  value = google_service_account.triton.name
}
