<h2 style="text-align: center;">BUILD THE CONTAINER</h2>

docker build -f Dockerfile -t fusion_tutorial .

---
<h2 style="text-align: center;">RUN THE CONTAINER</h2>

docker run -it -p 8888:8888 -p 6006:6006 -d -v $(pwd)/notebooks:/notebooks fusion_tutorial

---

#### References:
https://hands-on.cloud/custom-keras-docker-container/