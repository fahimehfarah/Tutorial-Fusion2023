<h2 style="text-align: center;">BUILD THE CONTAINER</h2>

run this command from Docker folder:
- source create_and_run_the_container.sh 

if you got permission denied follow this guide (linux user):
-  https://docs.docker.com/engine/install/linux-postinstall/

---
- Clean current docker with:
- docker stop $(docker ps -a -q)
- docker rm $(docker ps -a -q) 