import sys
def print_cost(start_time, end_time, cost, epoch, num_epochs, batch, num_batches):

    # Calculate the elapsed time in seconds
    training_time_seconds = end_time - start_time

    # Format the elapsed time in "HH:mm:ss" format
    hours = int(training_time_seconds // 3600)
    minutes = int((training_time_seconds % 3600) // 60)
    seconds = int(training_time_seconds % 60)

    epoch_percent = ((epoch + 1) / num_epochs) * 100.0
    batch_percent = ((batch + 1) / num_batches) * 100.0

    formatted_time = f"[@ {hours:02d}:{minutes:02d}:{seconds:02d}] Training {int(epoch_percent)}% (epoch: {epoch + 1}/{num_epochs}  batch: {batch + 1}/{num_batches}) "

    # Print the result with a fixed length of 20 characters
    print(f"{formatted_time:60}cost = {cost}")

def print_test_progress(accuracy, current, total, bar_length=20):
    progress = current / total
    num_bar_chars = int(round(bar_length * progress))
    bar = "#" * num_bar_chars + "-" * (bar_length - num_bar_chars)
    formatted_percentage = "{:.2f}".format(progress * 100)
    percent_display = "{:>6}%".format(formatted_percentage)

    formatted_accuracy = "{:.2f}/1".format(accuracy)
    accuracy_display = "{:>6}".format(formatted_accuracy)

    progress_bar = f"Accuracy: {accuracy_display}        Test Progress: {percent_display} [{bar}]"
    sys.stdout.write("\r" + progress_bar)
    sys.stdout.flush()