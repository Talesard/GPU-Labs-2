echo "CPU vs GPU"
app.exe 1000 0.00001 200 gpu 
app.exe 2000 0.00001 200 gpu
app.exe 5000 0.00001 200 gpu
app.exe 10000 0.00001 200 gpu
app.exe 1000 0.00001 200 cpu 
app.exe 2000 0.00001 200 cpu
app.exe 5000 0.00001 200 cpu
app.exe 10000 0.00001 200 cpu 

echo "Test float error?"
app.exe 100 0.00001 200 gpu
app.exe 200 0.00001 200 gpu
app.exe 1000 0.00001 200 gpu
app.exe 2000 0.00001 200 gpu
app.exe 3000 0.00001 200 gpu
app.exe 4000 0.00001 200 gpu
app.exe 5000 0.00001 200 gpu
app.exe 6000 0.00001 200 gpu
app.exe 7000 0.00001 200 gpu
app.exe 8000 0.00001 200 gpu
app.exe 9000 0.00001 200 gpu
app.exe 10000 0.00001 200 gpu
