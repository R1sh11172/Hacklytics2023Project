import logo from './logo.svg';
import plot1 from './my_plot1.png';
import plot2 from './my_plot2.png';
import plot3 from './my_plot3.png';
import plot4 from './my_plot4.png';
import plot5 from './my_plot5.png';
import plot6 from './my_plot6.png';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1> Stonks </h1>
        <img src = {plot1} className="plot1" alt="logo" />
        <img src = {plot2} className="plot2" alt="logo" />
        <img src = {plot3} className="plot3" alt="logo" />
        <img src = {plot4} className="plot4" alt="logo" />
        <img src = {plot5} className="plot5" alt="logo" />
        <img src = {plot6} className="plot6" alt="logo" />
        
      </header>
    </div>
  );
}

export default App;
