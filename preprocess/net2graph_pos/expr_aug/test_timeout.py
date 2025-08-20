import signal

class TimeoutException(Exception):
    """Custom exception to raise on a timeout"""
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Function execution timed out")

def function_with_timeout():
    # Set the timeout handler for the alarm signal
    signal.signal(signal.SIGALRM, timeout_handler)
    # Schedule an alarm in 10 seconds
    signal.alarm(3)

    try:
        # Your long-running function code here
        while True:
            print('Running')
            pass  # Simulate long-running task
    except:
        pass
        # print('Continue')
    finally:
        # Disable the alarm
        signal.alarm(0)

function_with_timeout()