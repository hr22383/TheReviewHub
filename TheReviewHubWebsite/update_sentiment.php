<?php
// Perform sentiment analysis using Python
$output = shell_exec('python sentiment_analysis.py');

// Display output on the web page
echo $output;
?>
