import logo from './logo.svg';
import plot1 from './my_plot1.png';
import plot2 from './my_plot2.png';
import plot3 from './my_plot3.png';

import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Uncovering Market Trends: A K-Means Clustering Approach</h1>
        <h2>By: Vivek, Aarin, Rishi, and Sai</h2>
        <p> Receive crucial information about the market </p>
        <img src = {plot1} className="plot1" alt="logo" />
        <p className="df_subtitle">Dataframe that shows clustering</p>
        <br></br>
        <br></br>
        <img src = {plot3} className="plot3" alt="logo" />
        <p className="volitality">Movement of Tesla vs Microsoft</p>
        <br></br>
        <br></br>
        <img src = {plot2} className="plot2" alt="logo" />
        <p className="cluster_subtitle">Visual cluster of stock data</p>
        
      </header>
    </div>
  );
}

export default App;
