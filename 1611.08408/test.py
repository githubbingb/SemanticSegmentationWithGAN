for step in range(3000):
    # adjust_learning_rate(optimizer, decay_rate=0.9, step=step)
    images, ground_truths = reader.next()
    print images, ground_truths