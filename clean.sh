rm -vrf ./__pycache__/
rm -vrf ./models/__pycache__/

rm -vrf ./tf_ops/nn_distance/*.pyc
rm -vrf ./tf_ops/nn_distance/__pycache__/
cd ./tf_ops/nn_distance
make clean
cd ../..

rm -vrf ./tf_ops/approxmatch/*.pyc
rm -vrf ./tf_ops/approxmatch/__pycache__/
cd ./tf_ops/approxmatch
make clean
cd ../..

rm -vrf ./utils/*.so
rm -vrf ./utils/__pycache__/


log_list=`find . -maxdepth 1 -name 'log*'`
if [ -n "$log_list" ]; then
  echo -e "\033[1m\033[95m"
  echo "Log Exist, Move or Remove It!"
  echo "$log_list"
  echo -e "\033[0m\033[0m"
fi

