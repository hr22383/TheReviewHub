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
	 
  </head>
  



<body>
<?php
include 'heade_user.php';
?>

<?php
  
include 'liveconfig.php';
//include 'localconfig.php';

 

 // Define variables and set them to empty values
$first_name = $last_name = $email = $password = $date_of_birth = "";

// Process form submission
if ($_SERVER["REQUEST_METHOD"] == "POST") {
  // Sanitize and validate input data (you can add more validation as needed)
  $first_name = sanitizeInput($_POST["inputFname"]);
  $last_name = sanitizeInput($_POST["inputLname"]);
  $email = sanitizeInput($_POST["inputEmail"]);
  $password = sanitizeInput($_POST["inputPassword"]);
  $date_of_birth = sanitizeInput($_POST["dateInput"]);
// Convert the input date to the appropriate format

$date_of_birth = DateTime::createFromFormat('d/m/Y', $date_of_birth)->format('Y-m-d');


  // Insert data into the database table
  $sql = "INSERT INTO user_new (first_name, last_name, email, password, date_of_birth, status, registration_date)
          VALUES ('$first_name', '$last_name', '$email', '$password', '$date_of_birth', 'Active', NOW())";


  if (mysqli_query($conn, $sql)) {
    echo "<script>showMessage();</script>";
    // Clear form input values
    $first_name = $last_name = $email = $password = $date_of_birth = "";
  } else {
    echo "Error: " . $sql . "<br>" . mysqli_error($conn);
  }
}

// Function to sanitize input data
function sanitizeInput($data) {
  $data = trim($data);
  $data = stripslashes($data);
  $data = htmlspecialchars($data);
  return $data;
}

// Close the database connection
mysqli_close($conn);
?>


<!--
Lower Header Section 
-->


	
	<form class="form-horizontal" method="POST" action="<?php echo htmlspecialchars($_SERVER["PHP_SELF"]); ?>">
    <h3>Your Personal Details</h3>
   

    <div class="control-group">
      <label class="control-label" for="inputFname">First name <sup class="sup">*</sup></label>
      <div class="controls">
        <input type="text" id="inputFname" name="inputFname" placeholder="First Name" value="<?php echo $first_name; ?>">
      </div>
    </div>

    <div class="control-group">
      <label class="control-label" for="inputLname">Last name <sup class="sup">*</sup></label>
      <div class="controls">
        <input type="text" id="inputLname" name="inputLname" placeholder="Last Name" value="<?php echo $last_name; ?>">
      </div>
    </div>

  <div class="control-group">
  <label class="control-label" for="inputEmail">Email <sup class="sup">*</sup></label>
  <div class="controls">
    <input type="text" id="inputEmail" name="inputEmail" placeholder="Email" value="<?php echo $email; ?>">
  </div>
</div>

  <div class="control-group">
  <label class="control-label">Password <sup class="sup">*</sup></label>
  <div class="controls">
    <input type="password" id="inputPassword" name="inputPassword" placeholder="Password">
  </div>
</div>

  <div class="control-group">
  <label class="control-label">Date of Birth <sup class="sup">*</sup></label>
  <div class="controls">
    <input type="text" id="dateInput" name="dateInput" placeholder="dd/mm/yyyy" value="<?php echo $date_of_birth; ?>">
  </div>
</div>

    <div class="control-group">
      <div class="controls">
        <input type="submit" name="submitAccount" value="Submit" class="exclusive shopBtn">
      </div>
    </div>
  </form>
</div>


	</form>
</div>





</div>
</div>
<!-- 
Clients 
-->


<!--
Footer
-->

  </body>
</html>
<script>
// JavaScript function to validate the date format
function validateDateFormat(input) {
  // Regular expression to match the "dd/mm/yyyy" format
  var regex = /^\d{2}\/\d{2}\/\d{4}$/;
  
  if (!regex.test(input.value)) {
    alert("Invalid date format. Please enter the date in dd/mm/yyyy format.");
    input.value = ""; // Clear the input field
    input.focus();    // Set focus back to the input field
  }
}

// Event listener to validate the date format when the input loses focus
var dateInput = document.getElementById("dateInput");
dateInput.addEventListener("blur", function() {
  validateDateFormat(this);
});
</script>