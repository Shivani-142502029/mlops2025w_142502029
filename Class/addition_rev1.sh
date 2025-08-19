echo -n "Enter first number : "
read a

echo -n "Enter second number : "
read b

sum=$(echo "$a+$b" | bc -l)
echo "Sum of $a and $b : $sum"
echo "Sum of $a and $b : $sum" >> output.txt

fact(){
	res=1
	for i in $(seq 1 $1)
 	do 
		res=$(($res*i))
	done
	echo "Factorial of $sum is $res"
	echo "Factorial of $sum is $res" >> output.txt
}

fact "$sum"
diff=$(echo "diff addition.sh addition_rev1.sh")
$diff >> output.txt
echo "Difference between two files : $diff"
