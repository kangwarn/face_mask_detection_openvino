FROM mmphego/intel-openvino
COPY requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt
RUN sudo apt-get update
RUN sudo apt-get -y install pciutils lshw clinfo
WORKDIR /opt/intel/openvino/install_dependencies
RUN sudo -E ./install_NEO_OCL_driver.sh
WORKDIR /app
# ENTRYPOINT ["jupyter-lab", "--ip=0.0.0.0"]
