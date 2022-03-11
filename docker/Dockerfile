FROM huggingface/transformers-pytorch-gpu

RUN pip3 install --upgrade pip
RUN pip3 install ipython
RUN pip3 install --upgrade transformers
RUN pip3 install openai
RUN pip3 install pandas
RUN pip3 install matplotlib
RUN pip3 install datasets
RUN mkdir -p /partisanbrain/transformers_cache
ENV TRANSFORMERS_CACHE=/partisanbrain/transformers_cache
# WORKDIR "/partisanbrain"
# RUN cd /partisanbrain

ENTRYPOINT ["/bin/bash"]
