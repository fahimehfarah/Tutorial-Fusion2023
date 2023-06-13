# get the config object
configuration = get_config()
# in-line figure when using Matplotlib
configuration.IPKernelApp.pylab = 'inline'
configuration.NotebookApp.ip = '*'
configuration.NotebookApp.allow_remote_access = True
# do not open a browser window by default when using notebooks
configuration.NotebookApp.open_browser = False
# No token. Always use jupyter over ssh tunnel
configuration.NotebookApp.token = ''
configuration.NotebookApp.notebook_dir = '/notebooks'
# Allow to run Jupyter from root user inside Docker container
configuration.NotebookApp.allow_root = True