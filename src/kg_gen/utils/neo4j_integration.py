"""
Neo4j integration utilities for uploading knowledge graphs to Neo4j.

This module provides functionality to upload generated knowledge graphs
to Neo4j databases, both local and cloud instances (Neo4j AuraDB).
"""

from typing import Optional, Dict, Any, List
import logging
from neo4j import GraphDatabase, Driver
from ..models import Graph

logger = logging.getLogger(__name__)


class Neo4jUploader:
    """Handles uploading knowledge graphs to Neo4j databases."""

    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        """
        Initialize Neo4j connection.

        Args:
            uri: Neo4j connection URI (e.g., 'bolt://localhost:7687' or 'neo4j+s://...' for AuraDB)
            username: Database username
            password: Database password
            database: Database name (default: 'neo4j')
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver: Optional[Driver] = None

    def connect(self) -> bool:
        """
        Establish connection to Neo4j database.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.driver = GraphDatabase.driver(
                self.uri, auth=(self.username, self.password)
            )
            # Test the connection
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            logger.info(f"Successfully connected to Neo4j at {self.uri}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False

    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def upload_graph(
        self,
        graph: Graph,
        graph_name: Optional[str] = None,
        clear_existing: bool = False,
        add_properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Upload a knowledge graph to Neo4j.

        Args:
            graph: The Graph object to upload
            graph_name: Optional name for the graph (used as a label)
            clear_existing: Whether to clear existing nodes/relationships first
            add_properties: Additional properties to add to nodes

        Returns:
            bool: True if upload successful, False otherwise
        """
        if not self.driver:
            logger.error("No active Neo4j connection. Call connect() first.")
            return False

        try:
            with self.driver.session(database=self.database) as session:
                # Clear existing data if requested
                if clear_existing:
                    session.run("MATCH (n) DETACH DELETE n")
                    logger.info("Cleared existing graph data")

                # Create nodes
                node_count = self._create_nodes(
                    session, graph, graph_name, add_properties
                )

                # Create relationships
                rel_count = self._create_relationships(session, graph, graph_name)

                logger.info(
                    f"Successfully uploaded graph: {node_count} nodes, {rel_count} relationships"
                )
                return True

        except Exception as e:
            logger.error(f"Failed to upload graph: {e}")
            return False

    def _create_nodes(
        self,
        session,
        graph: Graph,
        graph_name: Optional[str] = None,
        add_properties: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Create nodes in Neo4j from graph entities."""
        properties = add_properties or {}

        # Add graph name as a property if provided
        if graph_name:
            properties["graph_name"] = graph_name

        # Create nodes with labels
        labels = ["Entity"]
        if graph_name:
            labels.append(graph_name)

        label_str = ":".join(labels)

        query = f"""
        UNWIND $entities AS entity
        MERGE (n:Entity {{name: entity}})
        SET n += $properties
        """

        result = session.run(
            query, entities=list(graph.entities), properties=properties
        )
        return len(list(result))

    def _create_relationships(
        self, session, graph: Graph, graph_name: Optional[str] = None
    ) -> int:
        """Create relationships in Neo4j from graph relations."""
        rel_count = 0

        for subject, predicate, obj in graph.relations:
            # Create relationship with predicate as relationship type
            # Clean predicate name for Neo4j (replace spaces with underscores)
            rel_type = predicate.replace(" ", "_").replace("-", "_").upper()

            query = f"""
            MATCH (s:Entity {{name: $subject}})
            MATCH (o:Entity {{name: $object}})
            MERGE (s)-[r:{rel_type}]->(o)
            SET r.predicate = $predicate
            """

            if graph_name:
                query += "SET r.graph_name = $graph_name"

            session.run(
                query,
                subject=subject,
                object=obj,
                predicate=predicate,
                graph_name=graph_name,
            )
            rel_count += 1

        return rel_count

    def query_graph(
        self, cypher_query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query on the Neo4j database.

        Args:
            cypher_query: Cypher query string
            parameters: Query parameters

        Returns:
            List of result records
        """
        if not self.driver:
            logger.error("No active Neo4j connection. Call connect() first.")
            return []

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(cypher_query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []

    def get_graph_stats(self) -> Dict[str, int]:
        """Get basic statistics about the uploaded graph."""
        stats_query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN 
            count(DISTINCT n) as node_count,
            count(DISTINCT r) as relationship_count,
            count(DISTINCT labels(n)) as label_count
        """

        results = self.query_graph(stats_query)
        if results:
            return results[0]
        return {"node_count": 0, "relationship_count": 0, "label_count": 0}


def upload_to_neo4j(
    graph: Graph,
    uri: str,
    username: str,
    password: str,
    database: str = "neo4j",
    graph_name: Optional[str] = None,
    clear_existing: bool = False,
    add_properties: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Convenience function to upload a graph to Neo4j.

    Args:
        graph: The Graph object to upload
        uri: Neo4j connection URI
        username: Database username
        password: Database password
        database: Database name
        graph_name: Optional name for the graph
        clear_existing: Whether to clear existing data first
        add_properties: Additional properties to add to nodes

    Returns:
        bool: True if upload successful, False otherwise
    """
    uploader = Neo4jUploader(uri, username, password, database)

    try:
        if uploader.connect():
            success = uploader.upload_graph(
                graph, graph_name, clear_existing, add_properties
            )
            return success
        return False
    finally:
        uploader.close()


# Example usage and connection configurations
def get_aura_connection_config(
    instance_id: str, username: str, password: str, region: str = "us-east-1"
) -> Dict[str, str]:
    """
    Get connection configuration for Neo4j AuraDB.

    Args:
        instance_id: Your AuraDB instance ID
        username: Database username
        password: Database password
        region: AWS region (default: us-east-1)

    Returns:
        Dict with connection parameters
    """
    uri = f"neo4j+s://{instance_id}.databases.neo4j.io"
    return {"uri": uri, "username": username, "password": password, "database": "neo4j"}


def get_local_connection_config(
    host: str = "localhost",
    port: int = 7687,
    username: str = "neo4j",
    password: str = "password",
) -> Dict[str, str]:
    """
    Get connection configuration for local Neo4j instance.

    Args:
        host: Neo4j host
        port: Neo4j port
        username: Database username
        password: Database password

    Returns:
        Dict with connection parameters
    """
    uri = f"bolt://{host}:{port}"
    return {"uri": uri, "username": username, "password": password, "database": "neo4j"}
