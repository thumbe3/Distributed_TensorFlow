INP=$1
if [ "$1" = "" ];then
 INP="cluster2"
fi

source cluster_utils.sh
start_cluster syncTF.py $INP
