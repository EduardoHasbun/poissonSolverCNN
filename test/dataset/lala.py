    # Save q and xq to a txt file
    with open(os.path.join('generated', 'charges_data.txt'), 'w') as f:
        for i in range(nits):
            f.write(f"Sample {i}:\n")
            f.write("q = " + ", ".join(map(str, q_list[i])) + "\n")
            f.write("xq = [" + "; ".join(" ".join(map(str, x)) for x in xq_list[i]) + "]\n\n")
