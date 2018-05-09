// this is a comment

// dollar sign - jquery find the item
// when the webpage is ready, it will call the function
// creating a function in
// funciton - no name

var cities = {
  "Orlando": {
    url:'http://www.orlandotomiamishuttle.com/wp-content/uploads/2017/09/ORLANDO-3.jpg',
    country: 'Florida, U.S.',
    population: 35000 },

  "Munich": {
    url:'https://thtpoland.com/edu/wp-content/uploads/2018/01/Munich-High-Definition-Wallpapers.jpg',
    country: 'Germany',
    population: 35000 },

  "Paris": {
    url:'https://i.ytimg.com/vi/_FYKIhJZdaI/maxresdefault.jpg',
    country: 'France',
    population: 35000 },


  "Oslo":  {
    url:'https://d1bvpoagx8hqbg.cloudfront.net/originals/my-erasmus-experience-oslo-norway-beatrice-e183189bcd988cba7cd8932facaf99e8.jpg',
    country: 'Norway',
    population: 35000 },

  "Moscow": {
    url:'https://higherlogicdownload.s3.amazonaws.com/IMANET/0be307fc-98fd-412d-b879-ae9a90f110de/UploadedImages/Moscow.jpg',
    country: 'Russia',
    population: 35000 },


  "Hong Kong": {
    url:'http://www.secretflying.com/wp-content/uploads/2015/03/hong-kong-2.jpg',
    country: 'China',
    population: 35000 },



  "Dubai": {
    url:'https://www.birmingham.ac.uk/Images/News/dubai-image.jpg',
    country: 'United Arab Emirates',
    population: 35000 },


  "Sydney": {
    url:'https://lonelyplanetimages.imgix.net/mastheads/65830387.jpg?sharp=10&vib=20&w=1200',
    country: 'Australia',
    population: 35000 },


  "San Diego": {
    country: 'California, U.S.',
    population: 35000 },

  "Singapore": {
    country: 'Singapore',
    population: 35000 },

  "Venice": {
    country: 'Italy',
    population: 35000 },

  "St. Petersburg": {
    country: 'Russia',
    population: 35000 },

  "London": {
    country: 'England',
    population: 35000 },

  "Anaheim": {
    country: 'California, U.S',
    population: 35000 },

  "Los Angeles": {
    country: 'California, U.S',
    population: 35000 },

  "Washington DC": {
    country: 'Washington DC, U.S',
    population: 35000 },

  "Honolulu": {
    country: 'Hawaii, U.S',
    population: 35000 },

  "Toronto": {
    country: 'Canada',
    population: 35000 },


  "Budapest": {
    country: 'Hungary',
    population: 35000 },


  "Boston": {
    country: 'Massachusetts, U.S.',
      population: 35000 }

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
    console.log('country', cities[response.predictions.pred1].country);

    const country1 = cities[response.predictions.pred1].country;
    const country2 = cities[response.predictions.pred2].country;
    $('#city1-country').val(country1);
    $('#city2-country').val(country2);
    $('#recommender1').val(response.predictions.pred1);
    $('#recommender2').val(response.predictions.pred2);
  //
  });
});
