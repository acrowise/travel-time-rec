// this is a comment

// dollar sign - jquery find the item
// when the webpage is ready, it will call the function
// creating a function in
// funciton - no name

$(document).ready(function(){
  console.log('web page is ready');
  // async and await goes together
  $('#inference').click(async function(){
    console.log('button was clicked');

    const city = $('#cityselector').val() * 1;
    const user = $('#userselector').val() * 1;

    const data = {
      city,
      user
    };
    console.log(data)
    // browser sending the message to server who is listening
    const response = await $.ajax('/inference', {
      data: JSON.stringify(data),   // turning data into string
      method: 'post',               // method
      contentType: 'application/json'    //
    });

    console.log('response', response);
    $('#recommender').val(response.prediction);
  //
  });
});
