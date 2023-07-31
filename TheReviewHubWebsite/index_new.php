<!DOCTYPE html>
<html>
<head>
    <title>Sentiment Analysis</title>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
</head>
<body>
    <h1>Sentiment Analysis</h1>
    
    <?php
  
//include 'liveconfig.php';
include 'localconfig.php';

    // Create a new MySQL connection
    $conn = new mysqli($servername, $username, $password, $dbname);

    // Check the connection
    if ($conn->connect_error) {
        die("Connection failed: " . $conn->connect_error);
    }

    // Check if the form was submitted
    if ($_SERVER['REQUEST_METHOD'] === 'POST') {
        // Get the review from the form
        $review = $_POST['review'];

        // Call the Python script for sentiment analysis
        $command = escapeshellcmd('python sentiment_analysis.py ' . escapeshellarg($review));
        $output = shell_exec($command);
        $sentiment = trim($output);

        // Insert the review and sentiment into the database table
        $sql = "INSERT INTO sentiment (review, sent) VALUES ('$review', '$sentiment')";

        if ($conn->query($sql) === TRUE) {
            echo "Review inserted successfully.";
        } else {
            echo "Error: " . $sql . "<br>" . $conn->error;
        }

        // Display the result
        echo "<p><strong>Review:</strong> $review</p>";
        echo "<p><strong>Sentiment:</strong> $sentiment</p>";
    }

    // Close the database connection
    $conn->close();
    ?>
    
    <form method="post">
        <label for="review">Enter a review:</label>
        <textarea id="review" name="review"></textarea>
        <br>
        <input type="submit" value="Analyze">
    </form>
    
    <script>
        // AJAX request to handle the form submission without page refresh
        $('form').submit(function(event) {
            event.preventDefault();
            var review = $('#review').val();
            $.ajax({
                type: 'POST',
                url: 'index.php',
                data: { review: review },
                success: function(response) {
                    $('#result').html(response);
                }
            });
        });
    </script>
</body>
</html>
