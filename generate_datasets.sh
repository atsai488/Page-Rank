MAX_BLOCKS=100

format_count() {
    local value="$1"
    awk -v n="$value" 'BEGIN {
        if (n >= 1000000) {
            if (n % 1000000 == 0) printf "%dm", n / 1000000;
            else printf "%.1fm", n / 1000000;
        } else if (n >= 1000) {
            if (n % 1000 == 0) printf "%dk", n / 1000;
            else printf "%.1fk", n / 1000;
        } else {
            printf "%d", n;
        }
    }'
}

node_counts=(500000 1000000 1500000 2000000)
edge_counts=(20000000 40000000)

for num_nodes in "${node_counts[@]}"; do

    # Choose number of domains
    n_domains=100   # fixed

    # Compute subdomains so total blocks ≤ 2000
    subdomains_per_domain=$((MAX_BLOCKS / n_domains))

    # Ensure at least 1
    if (( subdomains_per_domain < 1 )); then
        subdomains_per_domain=1
    fi

    # Compute pages per subdomain to match node count
    total_hosts=$((n_domains * subdomains_per_domain))
    pages_per_subdomain=$((num_nodes / total_hosts))

    # Ensure exact match
    if (( num_nodes % total_hosts != 0 )); then
        echo "Skipping ${num_nodes} (not divisible by hosts=${total_hosts})"
        continue
    fi

    # Format values for filenames
    formatted_nodes=$(format_count "$num_nodes")

    for num_edges in "${edge_counts[@]}"; do
        avg_out_degree=$((num_edges / num_nodes))

        formatted_edges=$(format_count "$num_edges")

        python generate_synthetic_dataset.py \
            --edge-output "data/synth_${formatted_nodes}_${formatted_edges}.txt" \
            --metadata-output "data/synth_${formatted_nodes}_${formatted_edges}_nodes.csv" \
            --seed 42 \
            --n-domains "${n_domains}" \
            --min-subdomains "${subdomains_per_domain}" \
            --max-subdomains "${subdomains_per_domain}" \
            --min-pages-per-subdomain "${pages_per_subdomain}" \
            --max-pages-per-subdomain "${pages_per_subdomain}" \
            --degree-model fixed \
            --avg-out-degree "${avg_out_degree}" \
            --p-same-subdomain 0.92 \
            --p-same-domain-other-subdomain 0.07 \
            --allow-self-loops \
            --workers 32
    done
done

