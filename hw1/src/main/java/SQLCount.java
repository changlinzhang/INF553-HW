import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class SQLCount {

    public static class TokenizerMapper
            extends Mapper<Object, Text, Text, Text>{

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            String filePath = ((FileSplit)context.getInputSplit()).getPath().toString();
            String line = value.toString();
            if(filePath.contains("city-simple")){
                String[] tokens = line.split("\t");
                String cityID = tokens[0];
                String countryCode = tokens[2];
                int population = Integer.valueOf(tokens[4]);
                if (population >= 1000000)
                    context.write(new Text(countryCode), new Text("a:"+ cityID));
            }
            else if (filePath.contains("country")){
                String[] tokens = line.split("\t");
                String countryCode = tokens[0];
                String countryName = tokens[1];
                context.write(new Text(countryCode) , new Text("b:"+countryName));
            }
        }
    }

    public static class IntSumReducer
            extends Reducer<Text,Text,Text,Text> {
        public void reduce(Text key, Iterable<Text> values,
                           Context context
        ) throws IOException, InterruptedException {
            int sum = 0;
            String countryName = null;
            for (Text val : values) {
                String curVal = val.toString();
                if (curVal.startsWith("a:")) {
                    sum += 1;
                } else { // b:
                    countryName = val.toString().substring(2);
                }
            }
            if (sum >= 3)
                context.write(new Text(countryName), new Text(String.valueOf(sum)));
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        if (otherArgs.length < 2) {
            System.err.println("Usage: SQLCount <in> [<in>...] <out>");
            System.exit(2);
        }
        Job job = Job.getInstance(conf, "SQL count");
        job.setJarByClass(SQLCount.class);
        job.setMapperClass(TokenizerMapper.class);
//        job.setCombinerClass(IntSumReducer.class);
        job.setReducerClass(IntSumReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
//        job.setMapOutputKeyClass(Text.class);
//        job.setMapOutputValueClass(Text.class);
        for (int i = 0; i < otherArgs.length - 1; ++i) {
            FileInputFormat.addInputPath(job, new Path(otherArgs[i]));
        }
        FileOutputFormat.setOutputPath(job,
                new Path(otherArgs[otherArgs.length - 1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
