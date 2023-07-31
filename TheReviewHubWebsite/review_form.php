<!-- review_form.php -->
<div id="reviewPopup" class="popup">
    <div class="popup-content">
        <span class="close">&times;</span>
        <h2>Write a Review</h2>
        <form method="POST" action="save_review.php">
            <input type="hidden" name="product_id" id="product_id" value="">
            <div>
                <label for="review">Review:</label>
                <textarea name="review" id="review" rows="5" cols="40"></textarea>
            </div>
            <div>
                <label for="rating">Rating:</label>
                <select name="rating" id="rating">
                    <option value="1">1 Star</option>
                    <option value="2">2 Stars</option>
                    <option value="3">3 Stars</option>
                    <option value="4">4 Stars</option>
                    <option value="5">5 Stars</option>
                </select>
            </div>
            <div>
                <input type="submit" value="Submit">
            </div>
        </form>
    </div>
</div>
