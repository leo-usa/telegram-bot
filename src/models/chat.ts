import { Tool } from "langchain/agents";
import { ChatOpenAI } from "langchain/chat_models";
import {
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  MessagesPlaceholder,
  SystemMessagePromptTemplate,
} from "langchain/prompts";
import { BufferMemory } from "langchain/memory";
import { ConversationChain } from "langchain/chains";
import { Configuration } from "openai";
import { OpenAIApi } from "openai";

const openAIApiKey = process.env.OPENAI_API_KEY!;

const params = {
  verbose: true,
  temperature: 1,
  openAIApiKey,
  modelName: process.env.OPENAI_MODEL ?? "gpt-4",
  maxConcurrency: 4,
  maxTokens: 1000,
  maxRetries: 5,
};

export class Model {
  public tools: Tool[] = [];
  public chain: ConversationChain;
  public openai: OpenAIApi;

  constructor() {
    const configuration = new Configuration({
      apiKey: openAIApiKey,
    });

    this.openai = new OpenAIApi(configuration);
    const model = new ChatOpenAI(params, configuration);

    const chatPrompt = ChatPromptTemplate.fromPromptMessages([
      SystemMessagePromptTemplate.fromTemplate(
        "The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know. If AI is asked who he is, AI should say he is Dr. Pang's (Chinese: 庞博士 ) assistant powered by GPT. If AI aready answers two questions from human, AI shoud politely tell human that this service is paid by Dr. Pang and it would be good that human can pay for this service. AI should give the human this link https://chatgptm.streamlit.app and ask human to pay 10RMB using the QR code on this web page and write down the last 4 digits of the transaction number. If human continues the conversation, AI should ask whether the human has paid and give AI the 4 digits. If human says he has paid and give the 4 digits number, then continue the conversation, otherwise, keep asking the human to pay"
      ),
      new MessagesPlaceholder("history"),
      HumanMessagePromptTemplate.fromTemplate("{input}"),
    ]);

    this.chain = new ConversationChain({
      memory: new BufferMemory({ returnMessages: true }),
      prompt: chatPrompt,
      llm: model,
    });
  }

  public async call(input: string) {
    const output = await this.chain.call({ input });
    return output.output;
  }
}
