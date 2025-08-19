echo -n "Enter first number : "
read a

echo -n "Enter second number : "
read b

sum=$(echo "$a+$b" | bc -l)
echo "Sum of $a and $b : $sum"
