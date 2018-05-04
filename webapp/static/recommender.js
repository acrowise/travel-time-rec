// this is a comment

// dollar sign - jquery find the item
// when the webpage is ready, it will call the function
// creating a function in
// funciton - no name

var cities = {
  "Salzburg": {
    url:'https://travelpassionate.com/wp-content/uploads/2018/01/Austria-Rainbow-over-Salzburg-castle-min.jpg',
    country: 'Austria',
    population: 35000
  }
};

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
    console.log('response', response.predictions.pred1);
    console.log('response', response.predictions.pred2);
    console.log('city_dict', cities[response.predictions.pred1]);

    //const url = cities[response.prediction].url;
    // $('#city-url').attr('src', url);
    $('#recommender1').val(response.predictions.pred1);
    $('#recommender2').val(response.predictions.pred2);
  //
  });
});
