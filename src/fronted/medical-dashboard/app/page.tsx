'use client';

import { CardTitle, CardDescription, CardHeader, CardContent, Card } from "@/components/ui/card";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { useState, useEffect } from "react";

// Interfaces para tipado fuerte
interface MetricData {
  class: string;
  precision: number;
  recall: number;
  'f1-score': number;
  support: number;
}

// <<< NUEVO: Interfaz para los datos de la matriz de confusión
interface ConfusionRow {
  class: string;
  metric: 'True Positive' | 'False Positive' | 'True Negative' | 'False Negative';
  value: number;
}

// <<< NUEVO: Interfaz para los datos transformados de la matriz
interface TransformedConfusionData {
  [key: string]: {
    'True Positive'?: number;
    'False Positive'?: number;
    'True Negative'?: number;
    'False Negative'?: number;
  };
}

export default function DashboardPage() {
  const [metricsData, setMetricsData] = useState<MetricData[]>([]);
  // <<< NUEVO: Estado para los datos de la matriz de confusión
  const [confusionData, setConfusionData] = useState<TransformedConfusionData>({});
  
  const targetClasses = ['Cardiovascular', 'Neurological', 'Hepatorenal', 'Oncological'];

  useEffect(() => {
    // Cargar métricas de clasificación
    fetch('/data/classification_metrics.json')
      .then((response) => response.json())
      .then((data: MetricData[]) => {
        const filteredData = data.filter((d) => targetClasses.includes(d.class));
        setMetricsData(filteredData);
      });

    // <<< NUEVO: Cargar datos de la matriz de confusión
    fetch('/data/confusion_matrix_data.json')
      .then((response) => response.json())
      .then((data: ConfusionRow[]) => {
        // Transformamos el array en un objeto para un acceso más fácil
        const transformed = data.reduce((acc, row) => {
          if (!acc[row.class]) {
            acc[row.class] = {};
          }
          acc[row.class][row.metric] = row.value;
          return acc;
        }, {} as TransformedConfusionData);
        setConfusionData(transformed);
      });
  }, []);

  return (
    <div className="flex flex-col w-full min-h-screen bg-gray-900 text-white">
      <main className="flex min-h-[calc(100vh_-_theme(spacing.16))] flex-1 flex-col gap-4 p-4 md:gap-8 md:p-10">
        <div className="max-w-6xl w-full mx-auto flex flex-col gap-4">
          <div className="text-center">
            <h1 className="font-bold text-4xl">Informe de clasificación de la literatura médica</h1>
            <p className="text-gray-400 mt-2">Análisis de incrustaciones BERT con el clasificador MLP</p>
          </div>
          <div className="grid gap-6 md:grid-cols-2">
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle>Métricas de rendimiento del modelo</CardTitle>
                <CardDescription>Indicadores clave de rendimiento y mediciones de precisión.</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="w-full h-[300px]">
                  <ResponsiveContainer>
                    <BarChart data={metricsData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#4A5568" />
                      <XAxis dataKey="class" stroke="#A0AEC0" fontSize={12} />
                      {/* <<< CORRECCIÓN: Añadido el dominio de 0 a 1 */}
                      <YAxis stroke="#A0AEC0" domain={[0, 1]} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#2D3748', border: 'none' }} 
                        labelStyle={{ color: '#E2E8F0' }}
                      />
                      <Legend wrapperStyle={{ color: '#E2E8F0' }} />
                      <Bar dataKey="precision" fill="#8884d8" name="Precisión" />
                      <Bar dataKey="recall" fill="#82ca9d" name="Recall" />
                      <Bar dataKey="f1-score" fill="#ffc658" name="F1-Score" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle>Desglose de la matriz de confusión</CardTitle>
                <CardDescription>Resultados por categoría: Verdaderos/Falsos Positivos/Negativos.</CardDescription>
              </CardHeader>
              <CardContent>
                {/* <<< NUEVO: Lógica para renderizar los datos de la matriz */}
                <div className="grid grid-cols-2 gap-4 text-center">
                  {targetClasses.map((className) => (
                    <div key={className} className="p-2 bg-gray-700 rounded-lg">
                      <h4 className="font-bold text-sm mb-2">{className}</h4>
                      <div className="grid grid-cols-2 gap-1 text-xs">
                        <div className="p-1 bg-green-900/50 rounded">
                          <p className="font-semibold">TP</p>
                          <p>{confusionData[className]?.['True Positive'] ?? '...'}</p>
                        </div>
                        <div className="p-1 bg-red-900/50 rounded">
                          <p className="font-semibold">FP</p>
                          <p>{confusionData[className]?.['False Positive'] ?? '...'}</p>
                        </div>
                        <div className="p-1 bg-red-900/50 rounded">
                          <p className="font-semibold">FN</p>
                          <p>{confusionData[className]?.['False Negative'] ?? '...'}</p>
                        </div>
                        <div className="p-1 bg-blue-900/50 rounded">
                          <p className="font-semibold">TN</p>
                          <p>{confusionData[className]?.['True Negative'] ?? '...'}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </main>
    </div>
  );
}