import logo from "./logo.svg";
import plot1 from "./my_plot1.png";
import plot2 from "./my_plot2.png";
import plot3 from "./my_plot3.png";
import plot4 from "./my_plot4.png";
import plot5 from "./my_plot5.png";
import plot6 from "./my_plot6.png";
import "./App.css";
import { send_sms } from "./send_sms.js";
var phoneNumbers = [];
var i = 0;
function App() {
  
  return (
    <div className="App">
      <header className="App-header">
        <h1> Stonks </h1>
        <p>
          Receive crucial information about the market through a simple text
          message!
        </p>
        

        <img src={plot1} className="plot1" alt="logo" />
        <p className="cluster_subtitle">Cluster of stock data</p>
        <img src={plot2} className="plot2" alt="logo" />
        <p className="stock_volatility_subtitle">
          Volatility of Apple and Amazon stock prices
        </p>
        <img src={plot3} className="plot3" alt="logo" />
        <p className="stock_prices_subtitle">
          Prices of Apple and Amazon stock prices
        </p>
        <img src={plot4} className="plot4" alt="logo" />
        <p className="stock_volume_subtitle">
          Volume of Apple and Amazon stocks traded
        </p>
        <img src={plot5} className="plot5" alt="logo" />
        <p className="stock_opening_subtitle">Opening prices of Amazon stock</p>
        <img src={plot6} className="plot6" alt="logo" />
        <p className="stock_closing_subtitle">Closing prices of Amazon stock</p>
      </header>
    </div>
  );
}

export default App;
