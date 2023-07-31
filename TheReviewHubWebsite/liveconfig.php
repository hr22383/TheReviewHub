<?php

// Database configuration
$host = "sql100.infinityfree.com"; // Replace with your database host
$username = "if0_34355172"; // Replace with your database username
$password = "gbcIkmvbff"; // Replace with your database password
$database = "if0_34355172_reviewhub"; // Replace with your database name



// Create a connection
$conn = new mysqli($host, $username, $password, $database);

// Check the connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}
?>
