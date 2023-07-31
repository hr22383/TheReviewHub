
<?php




// Place your PHP login code here
if ($_SERVER["REQUEST_METHOD"] == "POST") {
	
include 'liveconfig.php';
//include 'localconfig.php';



    // Retrieve email and password from form submission
    $email = $_POST["email"];
    $password = $_POST["inputPassword"];

    // Sanitize the user input
    $email = mysqli_real_escape_string($conn, $email);
    $password = mysqli_real_escape_string($conn, $password);

    // Query the database to check if the email and password match
    $sql = "SELECT * FROM user_new WHERE email = '$email' and password = '$password'";
    $result = mysqli_query($conn, $sql);

    if (mysqli_num_rows($result) == 1) {
        if ($result) {
            // Retrieve the first name from the database result
            $row = mysqli_fetch_assoc($result);
            $first_name = $row['first_name'];

			// Clear all existing session data
			session_unset();

			// Destroy the session
			session_destroy();
			
			// Start a new session
			session_start();

            // Store the first name in a session variable
            $_SESSION['first_name'] = $first_name;
           

            // JavaScript redirection
            echo "<script>window.location.href = 'index.php';</script>";
            exit;
        } else {
            $loginError = "Invalid email or password.";
        }
    }

    // Close the database connection
    mysqli_close($conn);
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
 <style>
    /* Example CSS styles for the form */

    .form-horizontal {
      max-width: 400px;
      margin: 0 auto;
      padding: 20px;
      background-color: #f2f2f2;
      border: 1px solid #ccc;
      border-radius: 5px;
    }

    .control-group {
      margin-bottom: 15px;
    }

    .control-label {
      font-weight: bold;
      display: block;
    }

    .controls {
      margin-left: 150px;
    }

    input[type="text"],
    input[type="password"] {
      width: 200px;
      padding: 5px;
      border: 1px solid #ccc;
      border-radius: 3px;
    }

    input[type="submit"] {
      padding: 10px 20px;
      background-color: #333;
      color: #fff;
      border: none;
      border-radius: 3px;
      cursor: pointer;
    }

    input[type="submit"]:hover {
      background-color: #555;
    }

    .sup {
      color: red;
    }
  </style>
  
   <style>
    /* Example CSS styles for the form */
    /* CSS styles go here */

    .sup {
      color: red;
    }
  </style>
  <script>
    function showMessage() {
      alert("Record saved successfully!");
    }
  </script>
	 
  </style>
  
  
</head>


<body>

<?php
include 'heade_user.php';
?>


  <form class="form-horizontal" method="POST" action="<?php echo htmlspecialchars($_SERVER["PHP_SELF"]); ?>">
    <h3>Login</h3>

    <div class="control-group">
      <label class="control-label" for="email">Email <sup class="sup">*</sup></label>
      <div class="controls">
        <input type="text" id="email" name="email" placeholder="Email">
      </div>
    </div>

    <div class="control-group">
      <label class="control-label" for="inputPassword">Password <sup class="sup">*</sup></label>
      <div class="controls">
        <input type="password" id="inputPassword" name="inputPassword" placeholder="Password">
      </div>
    </div>

    <div class="control-group">
      <div class="controls">
        <input type="submit" name="login" value="Login">
      </div>
    </div>
    
    <div class="control-group">
      <div class="controls">
        <span>Not registered? <a href="register_user.php">Click here</a></span>
      </div>
    </div>
  </form>

</body>
</html>
