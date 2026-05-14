from sqlalchemy import create_engine, text

# Test Supabase Connection!
SUPABASE_CONNECTION_STRING = "postgresql://postgres:obstraatherealagent2027@db.ckjpcnkbroakxjjnyinp.supabase.co:5432/postgres"

print("Testing Supabase connection...")
try:
    engine = create_engine(SUPABASE_CONNECTION_STRING)
    with engine.connect() as conn:
        print("Connected to Supabase successfully!")
        
        # Test inserting a simple agent
        conn.execute(text("""
            INSERT INTO agents (name, framework, owner, token)
            VALUES ('TestAgent', 'test', 'TestOwner', 'test-token-123')
            ON CONFLICT (token) DO NOTHING
        """))
        conn.commit()
        print("Test agent inserted!")
        
        # Query agents
        result = conn.execute(text("SELECT * FROM agents"))
        agents = result.fetchall()
        print(f"Found {len(agents)} agents in Supabase!")
        for agent in agents:
            print(f"  - Agent: {agent[1]}")
            
except Exception as e:
    print(f"Error: {e}")
