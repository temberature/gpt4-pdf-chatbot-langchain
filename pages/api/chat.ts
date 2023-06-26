import type { NextApiRequest, NextApiResponse } from 'next';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { MyPineconeStore } from '@/utils/mypinecone';
import { makeChain } from '@/utils/makechain';
import { pinecone } from '@/utils/pinecone-client';
import { PINECONE_INDEX_NAME, PINECONE_NAME_SPACE } from '@/config/pinecone';
import { Configuration, OpenAIApi } from "openai";
import { ScoredVector } from '@pinecone-database/pinecone';

const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});
const openai = new OpenAIApi(configuration);

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse,
) {
  const { question, history } = req.body;

  console.log('question', question);

  //only accept post requests
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }

  if (!question) {
    return res.status(400).json({ message: 'No question in the request' });
  }
  // OpenAI recommends replacing newlines with spaces for best results
  const sanitizedQuestion = question.trim().replaceAll('\n', ' ');

  try {
    const ada = await openai.createEmbedding({
      model: "text-embedding-ada-002",
      input: sanitizedQuestion,
    });
    // console.log(ada);
    const embedding = ada.data.data[0].embedding;

    const index = pinecone.Index(PINECONE_INDEX_NAME);

    const queryRequest = {
      vector: embedding,
      topK: 10,
      includeValues: false,
      includeMetadata: true,
      filter: {
        // genre: { $in: ["comedy", "documentary", "drama"] },
      },
      namespace: PINECONE_NAME_SPACE,
    };
    const queryResponse = await index.query({ queryRequest });
    const matches: ScoredVector[] = queryResponse.matches || [];

    const sourceMatches = matches.filter(match => match.score !== undefined && match.score > 0.8).slice(0, 7);

    const context = sourceMatches
      .map((match, index) => {
        const text = (match.metadata as { text?: string }).text || "";
        return `引用片段${index + 1}: """${text}"""`;
      }).join('\n\n');



    const content = `你是一个有用的AI助手。使用以下上下文来回答最后的问题。
    如果你不知道答案，只需说你不知道即可。请勿试图编造答案和引用链接。
    让我们一步一步思考。
    
    ${context}
    
    问题: ${sanitizedQuestion}
    Markdown格式的有用答案，请标出引用来源，确保在引用后使用[数字]符号注明引用片段，不用再列出引用来源，格式清晰，越详细越好:`;
    console.log(content);
    const chatCompletion = await openai.createChatCompletion({
      model: 'gpt-3.5-turbo-16k',
      temperature: 0.7,
      messages: [{
        role: "user", content: content
      }],
    });
    // console.log(chatCompletion.data.choices[0].message);
    // declare response as an object with optional properties
    let response: { text?: string, sourceDocuments?: Array<any> } = {};

    response.text = chatCompletion.data.choices?.[0]?.message?.content;

    response.sourceDocuments = sourceMatches?.map(match => {
      // console.log(match.metadata);
      return {
        ...match,
        pageContent: (match.metadata as { text?: string })?.text || ""
      };
    });
    

    console.log('response', response);
    res.status(200).json(response);
  } catch (error: any) {
    console.log('error', error);
    res.status(500).json({ error: error.message || 'Something went wrong' });
  }
}
