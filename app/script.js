(function () {
    // Show loading screen immediately on page load
    function showInitialLoadingScreen() {
        const loadingOverlay = document.createElement('div');
        loadingOverlay.id = 'kg-gen-loading-overlay';

        Object.assign(loadingOverlay.style, {
            position: 'fixed',
            top: '0',
            left: '0',
            right: '0',
            bottom: '0',
            width: '100vw',
            height: '100vh',
            background: 'rgba(255, 255, 255, 0.95)',
            backdropFilter: 'blur(4px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: '1000001',
            padding: '1rem',
            pointerEvents: 'auto',
            overflow: 'hidden',
            fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
        });

        loadingOverlay.innerHTML = `
            <div class="loading-card" style="
                background: white;
                border-radius: 8px;
                padding: 2rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                text-align: center;
                max-width: 300px;
                width: 100%;
                margin: auto;
                pointer-events: none;
            ">
                <div class="loading-spinner" style="
                    width: 32px;
                    height: 32px;
                    border: 3px solid #e5e7eb;
                    border-top: 3px solid #3b82f6;
                    border-radius: 50%;
                    animation: kg-spinner-spin 1s linear infinite;
                    margin: 0 auto 1rem;
                "></div>
                <h3 style="
                    margin: 0 0 0.5rem;
                    font-size: 1.125rem;
                    font-weight: 600;
                    color: #111827;
                    word-break: break-word;
                ">Loading</h3>
                <p style="
                    margin: 0;
                    color: #6b7280;
                    font-size: 0.875rem;
                    word-break: break-word;
                ">Initializing Knowledge Graph Explorer...</p>
            </div>
            <style>
                @keyframes kg-spinner-spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            </style>
        `;

        document.body.appendChild(loadingOverlay);
    }

    showInitialLoadingScreen();

    const locale = navigator.language || 'en-US';

    function compareIgnoreCase(a, b) {
        const valueA = (a ?? '').toString();
        const valueB = (b ?? '').toString();
        const lowerA = valueA.toLocaleLowerCase(locale);
        const lowerB = valueB.toLocaleLowerCase(locale);
        if (lowerA < lowerB) {
            return -1;
        }
        if (lowerA > lowerB) {
            return 1;
        }
        if (valueA < valueB) {
            return -1;
        }
        if (valueA > valueB) {
            return 1;
        }
        return 0;
    }

    function sortedIgnoreCase(values) {
        if (!values) {
            return [];
        }
        if (values instanceof Set) {
            return Array.from(values).map(value => (value ?? '').toString()).sort(compareIgnoreCase);
        }
        return Array.from(values).map(value => (value ?? '').toString()).sort(compareIgnoreCase);
    }

    function ensureArray(value) {
        if (!value) {
            return [];
        }
        if (Array.isArray(value)) {
            return value;
        }
        if (value instanceof Set) {
            return Array.from(value);
        }
        if (typeof value === 'object') {
            return Object.values(value);
        }
        return [value];
    }

    function normalizeRelations(rawRelations) {
        return ensureArray(rawRelations)
            .map(entry => {
                if (!entry) {
                    return null;
                }
                if (Array.isArray(entry)) {
                    const [subject, predicate, object] = entry;
                    if (
                        subject === undefined ||
                        predicate === undefined ||
                        object === undefined
                    ) {
                        return null;
                    }
                    return [subject.toString(), predicate.toString(), object.toString()];
                }
                if (typeof entry === 'object') {
                    const subject = entry.subject ?? entry.source;
                    const predicate = entry.predicate ?? entry.relation ?? entry.label;
                    const object = entry.object ?? entry.target;
                    if (
                        subject === undefined ||
                        predicate === undefined ||
                        object === undefined
                    ) {
                        return null;
                    }
                    return [subject.toString(), predicate.toString(), object.toString()];
                }
                return null;
            })
            .filter(Boolean);
    }

    function normalizeClusterMap(rawClusters) {
        const map = new Map();
        if (!rawClusters) {
            return map;
        }
        if (rawClusters instanceof Map) {
            rawClusters.forEach((members, key) => {
                map.set(key.toString(), ensureArray(members).map(member => member.toString()));
            });
            return map;
        }
        const entries = Array.isArray(rawClusters)
            ? rawClusters
            : Object.entries(rawClusters);

        entries.forEach(entry => {
            if (!entry) {
                return;
            }
            if (Array.isArray(entry)) {
                const [key, members] = entry;
                if (key === undefined) {
                    return;
                }
                map.set(key.toString(), ensureArray(members).map(member => member.toString()));
            } else if (typeof entry === 'object' && entry.id !== undefined) {
                map.set(entry.id.toString(), ensureArray(entry.members).map(member => member.toString()));
            }
        });

        return map;
    }

    function normalizeClusterObject(rawClusters) {
        if (!rawClusters) {
            return {};
        }

        if (rawClusters instanceof Map) {
            const result = {};
            rawClusters.forEach((members, key) => {
                if (key === undefined || key === null) {
                    return;
                }
                result[key.toString()] = ensureArray(members).map(member => member.toString());
            });
            return result;
        }

        const entries = Array.isArray(rawClusters)
            ? rawClusters
            : Object.entries(rawClusters);

        const result = {};
        entries.forEach(entry => {
            if (!entry) {
                return;
            }
            if (Array.isArray(entry)) {
                const [key, members] = entry;
                if (key === undefined || key === null) {
                    return;
                }
                result[key.toString()] = ensureArray(members).map(member => member.toString());
            } else if (typeof entry === 'object' && entry.id !== undefined && entry.id !== null) {
                result[entry.id.toString()] = ensureArray(entry.members).map(member => member.toString());
            } else if (typeof entry === 'object') {
                Object.entries(entry).forEach(([key, members]) => {
                    if (key === undefined || key === null) {
                        return;
                    }
                    result[key.toString()] = ensureArray(members).map(member => member.toString());
                });
            }
        });

        return result;
    }

    function setStatus(message, type = 'info') {
        // no operation per now

    }

    function sanitizeGraphForBackend(rawGraph) {
        if (!rawGraph || typeof rawGraph !== 'object') {
            return rawGraph;
        }

        if (rawGraph.graph && typeof rawGraph.graph === 'object') {
            return {
                ...rawGraph,
                graph: sanitizeGraphForBackend(rawGraph.graph),
            };
        }

        const entityClustersSource = rawGraph.entity_clusters ?? rawGraph.entityClusters ?? rawGraph.entityclusters ?? {};
        const edgeClustersSource = rawGraph.edge_clusters ?? rawGraph.edgeClusters ?? rawGraph.edgeclusters ?? {};

        const normalizedEntityClusters = normalizeClusterObject(entityClustersSource);
        const normalizedEdgeClusters = normalizeClusterObject(edgeClustersSource);

        const payload = {
            ...rawGraph,
            entity_clusters: normalizedEntityClusters,
            edge_clusters: normalizedEdgeClusters,
        };

        if ('entityClusters' in payload) {
            payload.entityClusters = normalizedEntityClusters;
        }
        if ('entityclusters' in payload) {
            payload.entityclusters = normalizedEntityClusters;
        }
        if ('edgeClusters' in payload) {
            payload.edgeClusters = normalizedEdgeClusters;
        }
        if ('edgeclusters' in payload) {
            payload.edgeclusters = normalizedEdgeClusters;
        }

        return payload;
    }

    function normalizeGraphPayload(raw) {
        if (!raw || typeof raw !== 'object') {
            throw new Error('Graph JSON must be an object.');
        }
        if (raw.graph && typeof raw.graph === 'object') {
            return normalizeGraphPayload(raw.graph);
        }
        if (Array.isArray(raw.nodes) && Array.isArray(raw.edges) && raw.stats) {
            return raw;
        }

        const entities = ensureArray(raw.entities ?? raw.Nodes ?? raw.nodes ?? [])
            .map(entity => entity?.toString?.() ?? '')
            .filter(Boolean);

        const relationSource =
            raw.relations ??
            raw.Relations ??
            raw.relationships ??
            raw.triples ??
            raw.Triples ??
            raw.edges ??
            [];

        let relations = normalizeRelations(relationSource);
        if (
            !relations.length &&
            raw.relations &&
            typeof raw.relations === 'object' &&
            'data' in raw.relations
        ) {
            relations = normalizeRelations(raw.relations.data);
        }

        const entityClusters = normalizeClusterMap(
            raw.entity_clusters ?? raw.entityClusters ?? raw.entityclusters ?? []
        );

        const edgeClusters = normalizeClusterMap(
            raw.edge_clusters ?? raw.edgeClusters ?? raw.edgeclusters ?? []
        );

        return {
            entities,
            relations,
            entityClusters,
            edgeClusters,
        };
    }

    function hslToHex(h, s, l) {
        const hue = (h % 1 + 1) % 1;
        const saturation = Math.max(0, Math.min(1, s));
        const lightness = Math.max(0, Math.min(1, l));

        const a = saturation * Math.min(lightness, 1 - lightness);
        const f = n => {
            const k = (n + hue * 12) % 12;
            const color = lightness - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
            return Math.round(255 * color)
                .toString(16)
                .padStart(2, '0');
        };

        return `#${f(0)}${f(8)}${f(4)}`;
    }

    function stringToColor(label) {
        const text = (label ?? '').toString();
        let hash = 0;
        for (let index = 0; index < text.length; index += 1) {
            hash = Math.imul(31, hash) + text.charCodeAt(index);
            hash >>>= 0; // ensure unsigned 32-bit
        }
        const byte1 = hash & 0xff;
        const byte2 = (hash >>> 8) & 0xff;
        const byte3 = (hash >>> 16) & 0xff;
        const hue = byte1 / 255;
        const saturation = 0.55 + (byte2 / 255) * 0.3;
        const lightness = 0.45 + (byte3 / 255) * 0.25;
        return hslToHex(hue, saturation, lightness);
    }

    function buildViewModelLocally(rawGraph) {
        const graph = normalizeGraphPayload(rawGraph) ?? {};
        if (Array.isArray(graph.nodes) && Array.isArray(graph.edges) && graph.stats) {
            return graph;
        }

        const entitiesInput = ensureArray(graph.entities);
        const relationsInput = Array.isArray(graph.relations)
            ? graph.relations
            : ensureArray(graph.relations);

        const entityClustersMap = graph.entityClusters instanceof Map
            ? graph.entityClusters
            : normalizeClusterMap(graph.entityClusters);

        const edgeClustersMap = graph.edgeClusters instanceof Map
            ? graph.edgeClusters
            : normalizeClusterMap(graph.edgeClusters);

        const allEntities = new Set(entitiesInput);
        relationsInput.forEach(([subject, , object]) => {
            if (subject) {
                allEntities.add(subject);
            }
            if (object) {
                allEntities.add(object);
            }
        });

        const entities = sortedIgnoreCase(allEntities);

        const clusterView = [];
        const entityMemberToCluster = new Map();
        const nodeColorLookup = new Map();

        entityClustersMap.forEach((members, representative) => {
            const memberSet = new Set(ensureArray(members));
            memberSet.add(representative);
            const orderedMembers = sortedIgnoreCase(memberSet);
            const color = stringToColor(`entity::${representative}`);
            clusterView.push({
                id: representative,
                label: representative,
                members: orderedMembers,
                size: orderedMembers.length,
                color,
            });
            orderedMembers.forEach(member => {
                entityMemberToCluster.set(member, representative);
                nodeColorLookup.set(member, color);
            });
        });

        if (!clusterView.length) {
            entities.forEach(entity => {
                nodeColorLookup.set(entity, stringToColor(`entity::${entity}`));
            });
        } else {
            entities.forEach(entity => {
                if (!nodeColorLookup.has(entity)) {
                    nodeColorLookup.set(entity, stringToColor(`entity::${entity}`));
                }
            });
        }

        const edgeClusterView = [];
        const edgeMemberToCluster = new Map();
        const edgeColorLookup = new Map();

        edgeClustersMap.forEach((members, representative) => {
            const memberSet = new Set(ensureArray(members));
            memberSet.add(representative);
            const orderedMembers = sortedIgnoreCase(memberSet);
            const color = stringToColor(`edge::${representative}`);
            edgeClusterView.push({
                id: representative,
                label: representative,
                members: orderedMembers,
                size: orderedMembers.length,
                color,
            });
            orderedMembers.forEach(member => {
                edgeMemberToCluster.set(member, representative);
                edgeColorLookup.set(member, color);
            });
        });

        const degree = new Map();
        const indegree = new Map();
        const outdegree = new Map();
        const predicateCounts = new Map();
        const adjacency = new Map();
        const nodeNeighbors = new Map();
        const nodeEdges = new Map();

        const ensureNodeRecord = node => {
            if (!degree.has(node)) {
                degree.set(node, 0);
            }
            if (!indegree.has(node)) {
                indegree.set(node, 0);
            }
            if (!outdegree.has(node)) {
                outdegree.set(node, 0);
            }
            if (!adjacency.has(node)) {
                adjacency.set(node, new Set());
            }
            if (!nodeNeighbors.has(node)) {
                nodeNeighbors.set(node, new Set());
            }
            if (!nodeEdges.has(node)) {
                nodeEdges.set(node, { incoming: [], outgoing: [] });
            }
        };

        entities.forEach(entity => {
            ensureNodeRecord(entity);
        });

        const edgesView = [];
        relationsInput.forEach(([subject, predicate, object], index) => {
            const source = subject ?? '';
            const target = object ?? '';
            const edgePredicate = predicate ?? '';

            ensureNodeRecord(source);
            ensureNodeRecord(target);

            predicateCounts.set(
                edgePredicate,
                (predicateCounts.get(edgePredicate) || 0) + 1
            );

            degree.set(source, (degree.get(source) || 0) + 1);
            degree.set(target, (degree.get(target) || 0) + 1);
            outdegree.set(source, (outdegree.get(source) || 0) + 1);
            indegree.set(target, (indegree.get(target) || 0) + 1);

            adjacency.get(source).add(target);
            adjacency.get(target).add(source);
            nodeNeighbors.get(source).add(target);
            nodeNeighbors.get(target).add(source);

            const edgeId = `e${index}`;
            const cluster = edgeMemberToCluster.get(edgePredicate) || null;
            const color =
                edgeColorLookup.get(edgePredicate) ||
                stringToColor(`predicate::${edgePredicate}`);
            if (!edgeColorLookup.has(edgePredicate)) {
                edgeColorLookup.set(edgePredicate, color);
            }

            edgesView.push({
                id: edgeId,
                source,
                target,
                predicate: edgePredicate,
                cluster,
                color,
                tooltip: `${source} \u2014${edgePredicate}\u2192 ${target}`,
            });

            nodeEdges.get(source).outgoing.push(edgeId);
            nodeEdges.get(target).incoming.push(edgeId);
        });

        const isolatedEntities = entities.filter(
            entity => (degree.get(entity) || 0) === 0
        );

        const components = (() => {
            const visited = new Set();
            const result = [];
            entities.forEach(entity => {
                if (visited.has(entity)) {
                    return;
                }
                const queue = [entity];
                visited.add(entity);
                const members = [];
                while (queue.length) {
                    const current = queue.shift();
                    members.push(current);
                    (adjacency.get(current) || new Set()).forEach(neighbor => {
                        if (!visited.has(neighbor)) {
                            visited.add(neighbor);
                            queue.push(neighbor);
                        }
                    });
                }
                result.push({
                    size: members.length,
                    members: sortedIgnoreCase(members),
                });
            });
            result.sort((a, b) => {
                if (b.size !== a.size) {
                    return b.size - a.size;
                }
                const firstA = a.members[0] ?? '';
                const firstB = b.members[0] ?? '';
                return compareIgnoreCase(firstA, firstB);
            });
            return result;
        })();

        const nodesView = entities.map(entity => {
            const cluster = entityMemberToCluster.get(entity) || null;
            const nodeDegree = degree.get(entity) || 0;
            return {
                id: entity,
                label: entity,
                cluster,
                color: nodeColorLookup.get(entity) || '#64748b',
                degree: nodeDegree,
                indegree: indegree.get(entity) || 0,
                outdegree: outdegree.get(entity) || 0,
                isRepresentative: cluster ? cluster === entity : false,
                radius: 18 + Math.min(nodeDegree, 8) * 2,
                neighbors: sortedIgnoreCase(nodeNeighbors.get(entity) || []),
                edgeIds: nodeEdges.get(entity) || { incoming: [], outgoing: [] },
            };
        });

        const topEntities = nodesView
            .map(node => ({
                label: node.label,
                degree: node.degree,
                indegree: node.indegree,
                outdegree: node.outdegree,
                cluster: node.cluster,
            }))
            .sort((a, b) => {
                if (b.degree !== a.degree) {
                    return b.degree - a.degree;
                }
                return compareIgnoreCase(a.label, b.label);
            })
            .slice(0, 10);

        const topRelations = Array.from(predicateCounts.entries())
            .map(([predicate, count]) => ({
                predicate,
                count,
                cluster: edgeMemberToCluster.get(predicate) || null,
                color:
                    edgeColorLookup.get(predicate) ||
                    stringToColor(`predicate::${predicate}`),
            }))
            .sort((a, b) => {
                if (b.count !== a.count) {
                    return b.count - a.count;
                }
                return compareIgnoreCase(a.predicate, b.predicate);
            })
            .slice(0, 10);

        const totalDegree = entities.reduce(
            (sum, entity) => sum + (degree.get(entity) || 0),
            0
        );

        const stats = {
            entities: entities.length,
            relations: edgesView.length,
            relationTypes: predicateCounts.size,
            entityClusters: clusterView.length,
            edgeClusters: edgeClusterView.length,
            isolatedEntities: isolatedEntities.length,
            components: components.length,
            averageDegree: entities.length
                ? Math.round((totalDegree / entities.length) * 100) / 100
                : 0,
            density:
                entities.length > 1
                    ? Math.round(
                        (edgesView.length /
                            (entities.length * (entities.length - 1)) +
                            Number.EPSILON) *
                        1000
                    ) / 1000
                    : 0,
        };

        const relationRecords = edgesView.map((edge, index) => {
            const relation = relationsInput[index] || [];
            return {
                source: relation[0] ?? edge.source,
                predicate: relation[1] ?? edge.predicate,
                target: relation[2] ?? edge.target,
                edgeId: edge.id,
                color: edge.color,
            };
        });

        return {
            nodes: nodesView,
            edges: edgesView,
            clusters: clusterView,
            edgeClusters: edgeClusterView,
            topEntities,
            topRelations,
            stats,
            isolatedEntities,
            components,
            relations: relationRecords,
        };
    }

    const viewer = document.getElementById('viewer');
    const viewerWrapper = document.querySelector('.viewer-wrapper');
    const floatingActions = document.getElementById('floatingActions');
    const downloadButton = document.getElementById('downloadGraph');
    const refreshButton = document.getElementById('refreshView');

    let graphFileInput = document.getElementById('graphFile');
    let graphDropZone = document.getElementById('graphDropZone');
    let exampleSelect = document.getElementById('exampleGraph');
    let exampleLink = document.getElementById('exampleLink');
    let exampleStatus = document.getElementById('exampleStatus');
    let modalOpenGraphViewer = null;

    let apiKeyInput = document.getElementById('apiKey');
    let modelSelect = document.getElementById('model');
    let chunkSizeInput = document.getElementById('chunkSize');
    let temperatureInput = document.getElementById('temperature');
    let clusterToggle = document.getElementById('clusterToggle');
    let retrievalModelSelect = document.getElementById('retrievalModel');
    let contextInput = document.getElementById('context');
    let sourceText = document.getElementById('sourceText');
    let textFileInput = document.getElementById('textFile');

    let generateButton = document.getElementById('generateButton');
    let clearTextButton = document.getElementById('clearTextButton');

    console.log('[kg-gen] Element refs:', {
        apiKeyInput: !!apiKeyInput,
        modelSelect: !!modelSelect,
        chunkSizeInput: !!chunkSizeInput,
        temperatureInput: !!temperatureInput,
        clusterToggle: !!clusterToggle,
        retrievalModelSelect: !!retrievalModelSelect,
        contextInput: !!contextInput,
        viewer: !!viewer,
        viewerWrapper: !!viewerWrapper,
        floatingActions: !!floatingActions
    });

    const refreshCallbacks = [];
    const exampleMetadata = new Map();

    // localStorage keys for caching form inputs
    const CACHE_KEYS = {
        apiKey: 'kg-gen-api-key',
        model: 'kg-gen-model',
        chunkSize: 'kg-gen-chunk-size',
        temperature: 'kg-gen-temperature',
        cluster: 'kg-gen-cluster',
        retrievalModel: 'kg-gen-retrieval-model',
        context: 'kg-gen-context'
    };

    // Load cached form values from localStorage
    function loadCachedInputs() {
        console.log('[kg-gen] Loading cached inputs...');
        try {
            // Load API key
            const cachedApiKey = localStorage.getItem(CACHE_KEYS.apiKey);
            console.log('[kg-gen] Cached API key exists:', !!cachedApiKey);
            if (cachedApiKey && apiKeyInput) {
                apiKeyInput.value = cachedApiKey;
                console.log('[kg-gen] API key loaded');
            }

            // Load model
            const cachedModel = localStorage.getItem(CACHE_KEYS.model);
            console.log('[kg-gen] Cached model:', cachedModel);
            if (cachedModel && modelSelect) {
                modelSelect.value = cachedModel;
                console.log('[kg-gen] Model loaded:', modelSelect.value);
            }

            // Load chunk size
            const cachedChunkSize = localStorage.getItem(CACHE_KEYS.chunkSize);
            if (cachedChunkSize && chunkSizeInput) {
                chunkSizeInput.value = cachedChunkSize;
            }

            // Load temperature
            const cachedTemperature = localStorage.getItem(CACHE_KEYS.temperature);
            if (cachedTemperature && temperatureInput) {
                temperatureInput.value = cachedTemperature;
            }

            // Load cluster toggle
            const cachedCluster = localStorage.getItem(CACHE_KEYS.cluster);
            if (cachedCluster !== null && clusterToggle) {
                clusterToggle.checked = cachedCluster === 'true';
            }

            // Load retrieval model
            const cachedRetrievalModel = localStorage.getItem(CACHE_KEYS.retrievalModel);
            console.log('[kg-gen] Cached retrieval model:', cachedRetrievalModel);
            if (cachedRetrievalModel && retrievalModelSelect) {
                retrievalModelSelect.value = cachedRetrievalModel;
                console.log('[kg-gen] Retrieval model loaded:', retrievalModelSelect.value);
            }

            // Load context
            const cachedContext = localStorage.getItem(CACHE_KEYS.context);
            if (cachedContext && contextInput) {
                contextInput.value = cachedContext;
            }

            console.log('[kg-gen] Cached inputs loaded successfully');
        } catch (error) {
            console.warn('[kg-gen] Failed to load cached inputs:', error);
        }
    }

    // Save form input to localStorage
    function saveCachedInput(key, value) {
        try {
            if (value !== undefined && value !== null) {
                localStorage.setItem(key, String(value));
            } else {
                localStorage.removeItem(key);
            }
        } catch (error) {
            console.warn('[kg-gen] Failed to save cached input:', key, error);
        }
    }

    // Setup input change listeners for caching
    function setupInputCaching() {
        // API key caching
        if (apiKeyInput) {
            apiKeyInput.addEventListener('input', () => {
                saveCachedInput(CACHE_KEYS.apiKey, apiKeyInput.value);
            });
        }

        // Model caching
        if (modelSelect) {
            modelSelect.addEventListener('change', () => {
                saveCachedInput(CACHE_KEYS.model, modelSelect.value);
            });
        }

        // Chunk size caching
        if (chunkSizeInput) {
            chunkSizeInput.addEventListener('input', () => {
                saveCachedInput(CACHE_KEYS.chunkSize, chunkSizeInput.value);
            });
        }

        // Temperature caching
        if (temperatureInput) {
            temperatureInput.addEventListener('input', () => {
                saveCachedInput(CACHE_KEYS.temperature, temperatureInput.value);
            });
        }

        // Cluster toggle caching
        if (clusterToggle) {
            clusterToggle.addEventListener('change', () => {
                saveCachedInput(CACHE_KEYS.cluster, clusterToggle.checked);
            });
        }

        // Retrieval model caching
        if (retrievalModelSelect) {
            retrievalModelSelect.addEventListener('change', () => {
                saveCachedInput(CACHE_KEYS.retrievalModel, retrievalModelSelect.value);
            });
        }

        // Context caching
        if (contextInput) {
            contextInput.addEventListener('input', () => {
                saveCachedInput(CACHE_KEYS.context, contextInput.value);
            });
        }

        // Enhanced mutual exclusion with visual feedback
        const textFileHint = document.getElementById('textFileHint');

        window.updateInputStates = function updateInputStates() {
            const hasText = sourceText && sourceText.value.trim();
            const hasFile = textFileInput && textFileInput.files && textFileInput.files.length > 0;
            const hasContent = hasText || hasFile;

            // Update input states
            if (sourceText) {
                sourceText.disabled = hasFile;
            }
            if (textFileInput) {
                textFileInput.disabled = hasText;
            }

            // Update hint visibility
            if (textFileHint) {
                if (hasText) {
                    textFileHint.classList.add('show');
                } else {
                    textFileHint.classList.remove('show');
                }
            }
        }

        if (textFileInput) {
            textFileInput.addEventListener('change', updateInputStates);
        }

        if (sourceText) {
            sourceText.addEventListener('input', updateInputStates);
        }

        // Initial state
        updateInputStates();
    }

    const modelDefaultTemperature = new Map([
        ['openai/gpt-5', 1.0],
        ['openai/gpt-5-nano', 1.0],
        ['openai/gpt-5-mini', 1.0],
        ['openai/gpt-5-large', 1.0],
        ['openai/gpt-4.1', 0.0],
        ['openai/gpt-4.1-mini', 0.0],
        ['openai/gpt-4.1-nano', 0.0],
        ['openai/gpt-4o', 0.0],
        ['openai/gpt-4o-mini', 0.0],
        ['openai/gpt-4o-mini-omni-math', 0.0],
    ]);

    function inferDefaultTemperature(modelId) {
        if (!modelId) {
            return 0;
        }
        if (modelDefaultTemperature.has(modelId)) {
            return modelDefaultTemperature.get(modelId);
        }
        if (modelId.includes('gpt-5')) {
            return 1.0;
        }
        return 0.0;
    }

    function applyDefaultTemperature(modelId, userTriggered = false) {
        if (!temperatureInput) {
            return;
        }
        const currentValue = temperatureInput.value.trim();
        if (!userTriggered && currentValue) {
            return;
        }
        const temperature = inferDefaultTemperature(modelId);
        temperatureInput.value = temperature.toString();
    }

    let templateHtml = null;
    let activeUrl = null;
    let lastGraphPayload = null;
    let lastViewModel = null;
    let isGenerating = false;
    let hasLoadedGraph = false;

    function resetViewer() {
        if (activeUrl) {
            URL.revokeObjectURL(activeUrl);
            activeUrl = null;
        }
        viewer.setAttribute('hidden', 'hidden');
        viewer.removeAttribute('src');
        floatingActions.setAttribute('hidden', 'hidden');
        refreshCallbacks.length = 0;
        hasLoadedGraph = false;
        if (viewerWrapper) {
            viewerWrapper.classList.remove('graph-loaded');
        }
    }

    // Global loading system - always shows full-screen overlay
    function showGlobalLoading(title, message) {
        showLoadingInPlaceholder(title, message);
    }

    function hideGlobalLoading() {
        hideLoadingInPlaceholder();
    }

    // Legacy functions for backward compatibility - now use global loading
    function showLoadingInViewer(title, message) {
        showGlobalLoading(title, message);
    }

    function hideLoadingInViewer() {
        hideGlobalLoading();
    }

    function showLoadingInPlaceholder(title, message) {
        // Remove any existing loading overlay first
        const existingOverlay = document.getElementById('kg-gen-loading-overlay');
        if (existingOverlay) {
            existingOverlay.remove();
        }

        // Create loading overlay that covers entire screen
        const loadingOverlay = document.createElement('div');
        loadingOverlay.id = 'kg-gen-loading-overlay';

        // Set all styles directly on the element to ensure they're applied
        Object.assign(loadingOverlay.style, {
            position: 'fixed',
            top: '0',
            left: '0',
            right: '0',
            bottom: '0',
            width: '100vw',
            height: '100vh',
            background: 'rgba(255, 255, 255, 0.95)',
            backdropFilter: 'blur(4px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: '1000001',
            padding: '1rem',
            pointerEvents: 'auto',
            overflow: 'hidden',
            fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
        });

        loadingOverlay.innerHTML = `
            <div class="loading-card" style="
                background: white;
                border-radius: 8px;
                padding: 2rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                text-align: center;
                max-width: 300px;
                width: 100%;
                margin: auto;
                pointer-events: none;
            ">
                <div class="loading-spinner" style="
                    width: 32px;
                    height: 32px;
                    border: 3px solid #e5e7eb;
                    border-top: 3px solid #3b82f6;
                    border-radius: 50%;
                    animation: kg-spinner-spin 1s linear infinite;
                    margin: 0 auto 1rem;
                "></div>
                <h3 style="
                    margin: 0 0 0.5rem;
                    font-size: 1.125rem;
                    font-weight: 600;
                    color: #111827;
                    word-break: break-word;
                ">${title}</h3>
                <p style="
                    margin: 0;
                    color: #6b7280;
                    font-size: 0.875rem;
                    word-break: break-word;
                ">${message}</p>
            </div>
            <style>
                @keyframes kg-spinner-spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            </style>
        `;

        // Add event listeners to prevent any clicks from going through
        loadingOverlay.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation();
        }, true);

        loadingOverlay.addEventListener('mousedown', (e) => {
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation();
        }, true);

        loadingOverlay.addEventListener('keydown', (e) => {
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation();
        }, true);

        // Insert at the very end of the body
        document.body.appendChild(loadingOverlay);

    }

    function hideLoadingInPlaceholder() {
        // Remove the full-screen loading overlay
        const loadingOverlay = document.getElementById('kg-gen-loading-overlay');
        if (loadingOverlay) {
            loadingOverlay.remove();
        }
    }

    function confirmGraphReplacement(action) {
        if (!hasLoadedGraph || !isGenerating) {
            return true;
        }
        return confirm(
            'A graph is currently loaded. Proceeding will replace it.\n\n' +
            'Continue with ' + action + '?'
        );
    }

    function prepareDownload(graphJson) {
        if (!graphJson) {
            downloadButton.setAttribute('hidden', 'hidden');
            refreshButton.setAttribute('hidden', 'hidden');
            floatingActions.setAttribute('hidden', 'hidden');
            return;
        }
        lastGraphPayload = graphJson;
        downloadButton.removeAttribute('hidden');
        refreshButton.removeAttribute('hidden');
        floatingActions.removeAttribute('hidden');
    }

    function updateExampleLink(meta) {
        if (meta && meta.wiki_url) {
            exampleLink.href = meta.wiki_url;
            const label = meta.title ? `View ${meta.title} on Wikipedia` : 'View on Wikipedia';
            exampleLink.textContent = label;
            exampleLink.removeAttribute('hidden');
        } else {
            exampleLink.setAttribute('hidden', 'hidden');
            exampleLink.href = '#';
            exampleLink.textContent = 'View on Wikipedia';
        }
    }

    async function loadInitialExampleGraph() {

        document.getElementById('errorGraphMessage').classList.remove('show');
        showFloatingActions();
        try {
            const response = await fetch('/api/examples');
            const items = await response.json();
            if (!response.ok) {
                throw new Error(Array.isArray(items?.detail) ? items.detail.join('; ') : items?.detail || 'Failed to load sample list');
            }

            items.forEach(item => {
                if (!item || !item.slug) {
                    return;
                }
                exampleMetadata.set(item.slug, item);
            });

            if (exampleMetadata.size === 0) {
                console.error('[kg-gen] No examples found');
                return;
            }

            // Auto-select and load the first example
            if (items.length > 0) {
                const firstExample = items[0];
                console.log('[kg-gen] Auto-loading first example:', firstExample);
                // Show loading screen immediately
                showLoadingInViewer('Loading Example', 'Loading sample graph...');

                // Automatically load the first example
                const meta = exampleMetadata.get(firstExample.slug);
                // updateExampleLink(meta);
                const title = meta?.title || firstExample.slug;
                setStatus(`Loading example graph: ${title}...`);

                if (!hasLoadedGraph) {
                    resetViewer();
                }

                fetch(`/api/examples/${firstExample.slug}`)
                    .then(async response => {
                        let payload;
                        try {
                            payload = await response.json();
                        } catch (parseError) {
                            if (response.ok) {
                                throw new Error('Example payload is not valid JSON');
                            }
                            throw new Error(`Request failed (${response.status})`);
                        }

                        if (!response.ok) {
                            const message = payload?.detail || payload?.error || `Failed to load example (${response.status})`;
                            throw new Error(message);
                        }

                        return handleGraphData(payload);
                    })
                    .then(() => {
                        // exampleStatus.textContent = `Loaded ${title}.`;
                    })
                    .catch(error => {
                        console.error('[kg-gen] Failed to load example graph', error);
                        setStatus(`Failed to load example '${title}': ${error.message}`, 'error');
                        // exampleStatus.textContent = 'Could not load the selected sample.';
                        hideLoadingInViewer();
                    })
                    .finally(() => {
                    });
            }
        } catch (error) {
            document.getElementById('errorGraphMessage').classList.add('show');
            hideFloatingActions();
            console.error('[kg-gen] Failed to load example graphs', error);
            setStatus(`Failed to load example graphs: ${error.message}`, 'error');
            hideLoadingInViewer();
            resetViewer();
        }
     
    }

    async function loadExamples() {
        if (!exampleSelect) {
            return;
        }

        exampleSelect.disabled = true;
        exampleMetadata.clear();
        exampleSelect.innerHTML = '';
        const loadingOption = document.createElement('option');
        loadingOption.value = '';
        loadingOption.textContent = 'Loading samples...';
        exampleSelect.appendChild(loadingOption);
        exampleStatus.textContent = 'Loading sample graphs...';
        updateExampleLink(null);

        try {
            const response = await fetch('/api/examples');
            const items = await response.json();
            if (!response.ok) {
                throw new Error(Array.isArray(items?.detail) ? items.detail.join('; ') : items?.detail || 'Failed to load sample list');
            }

            if (!Array.isArray(items) || !items.length) {
                const emptyOption = document.createElement('option');
                emptyOption.value = '';
                emptyOption.textContent = 'No samples available';
                exampleSelect.innerHTML = '';
                exampleSelect.appendChild(emptyOption);
                exampleStatus.textContent = 'Sample graphs are not available right now.';
                return;
            }

            exampleSelect.innerHTML = '';
            const placeholderOption = document.createElement('option');
            placeholderOption.value = '';
            placeholderOption.textContent = 'Pick a sample graph...';
            exampleSelect.appendChild(placeholderOption);

            items.forEach(item => {
                if (!item || !item.slug) {
                    return;
                }
                exampleMetadata.set(item.slug, item);
                const option = document.createElement('option');
                option.value = item.slug;
                option.textContent = item.title || item.slug;
                exampleSelect.appendChild(option);
            });

            if (exampleMetadata.size === 0) {
                const fallbackOption = document.createElement('option');
                fallbackOption.value = '';
                fallbackOption.textContent = 'Samples unavailable';
                exampleSelect.innerHTML = '';
                exampleSelect.appendChild(fallbackOption);
                exampleStatus.textContent = 'Sample graphs are not available right now.';
                return;
            }

            exampleSelect.disabled = false;
            exampleStatus.textContent = 'Select an example to load it instantly.';

            // Auto-select and load the first example
            if (items.length > 0) {
                const firstExample = items[0];
                console.log('[kg-gen] Auto-loading first example:', firstExample);
                exampleSelect.value = firstExample.slug;
                // Automatically load the first example
                const meta = exampleMetadata.get(firstExample.slug);
                updateExampleLink(meta);
                const title = meta?.title || firstExample.slug;
                exampleStatus.textContent = `Loaded ${title}.`;
                setStatus(`Loading example graph: ${title}...`);
            }
        } catch (error) {
            console.error('[kg-gen] Failed to load example graphs', error);
            exampleSelect.innerHTML = '';
            const fallbackOption = document.createElement('option');
            fallbackOption.value = '';
            fallbackOption.textContent = 'Samples unavailable';
            exampleSelect.appendChild(fallbackOption);
            exampleStatus.textContent = 'Could not load sample graphs.';
        }
    }

    function extractSummary(viewModel) {
        if (!viewModel || !viewModel.stats) {
            return 'Graph rendered.';
        }
        const stats = viewModel.stats;
        const summary = [];
        if (typeof stats.entities === 'number') summary.push(`${stats.entities} entities`);
        if (typeof stats.relations === 'number') summary.push(`${stats.relations} relations`);
        if (typeof stats.relationTypes === 'number') summary.push(`${stats.relationTypes} relation types`);
        return `Graph ready: ${summary.join(', ')}.`;
    }

    function sanitizeJson(jsonValue) {
        return JSON.stringify(jsonValue, null, 2).replace(/<\/script>/gi, '<\\/script>');
    }

    async function loadTemplate() {
        if (templateHtml) {
            return templateHtml;
        }
        setStatus('Loading visualization template...');
        try {
            const response = await fetch('/template');
            if (!response.ok) {
                throw new Error('Template fetch failed');
            }
            templateHtml = await response.text();
            setStatus('Template ready. Drop or generate a graph.');
        } catch (error) {
            setStatus(`Failed to load template: ${error.message}`, 'error');
            throw error;
        }
        return templateHtml;
    }

    async function renderView(viewModel, graphForDownload) {
        console.info('[kg-gen] Rendering view model');
        showLoadingInViewer('Rendering Graph', 'Building visualization...');
        await loadTemplate();
        const html = templateHtml.replace('<!--DATA-->', `\n${sanitizeJson(viewModel)}\n`);
        if (activeUrl) {
            URL.revokeObjectURL(activeUrl);
        }
        activeUrl = URL.createObjectURL(new Blob([html], { type: 'text/html' }));

        let fallbackTimer = null;

        function cleanupLoadHandlers() {
            viewer.removeEventListener('load', onLoadHandler);
            viewer.removeEventListener('error', onErrorHandler);
        }

        function onLoadHandler() {
            if (fallbackTimer !== null) {
                clearTimeout(fallbackTimer);
                fallbackTimer = null;
            }
            cleanupLoadHandlers();
            setTimeout(() => {
                hideLoadingInViewer();
            }, 500);
        }

        function onErrorHandler() {
            if (fallbackTimer !== null) {
                clearTimeout(fallbackTimer);
                fallbackTimer = null;
            }
            cleanupLoadHandlers();
            hideLoadingInViewer();
        }

        viewer.addEventListener('load', onLoadHandler, { once: true });
        viewer.addEventListener('error', onErrorHandler, { once: true });

        fallbackTimer = setTimeout(() => {
            cleanupLoadHandlers();
            fallbackTimer = null;
            hideLoadingInViewer();
        }, 5000);

        viewer.src = activeUrl;
        viewer.removeAttribute('hidden');
        if (viewerWrapper) {
            viewerWrapper.classList.add('graph-loaded');
        }
        prepareDownload(graphForDownload || viewModel);
        lastViewModel = viewModel;
        setStatus(extractSummary(viewModel), 'success');
        refreshCallbacks.length = 0;
        refreshCallbacks.push(() => renderView(lastViewModel, lastGraphPayload));
        hasLoadedGraph = true;

        // Notify sidebar manager about the new graph data
        if (window.sidebarManager) {
            setTimeout(() => {
                window.sidebarManager.handleGraphDataReady(viewModel);
            }, 500); // Small delay to ensure iframe is ready
        }
    }

    function readFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = event => resolve(event.target?.result ?? '');
            reader.onerror = () => reject(new Error('File read failed'));
            reader.readAsText(file);
        });
    }

    async function handleGraphData(rawJson) {
        console.log('[kg-gen] handleGraphData called with:', rawJson);
        if (!rawJson) {
            throw new Error('Empty JSON payload');
        }

        const isViewModel = rawJson.nodes && rawJson.edges && rawJson.stats;
        if (isViewModel) {
            console.info('[kg-gen] Detected precomputed view model');
            setStatus('Rendering supplied visualization...');
            await renderView(rawJson, rawJson);
            return;
        }

        setStatus('Preparing graph data...');
        console.info('[kg-gen] Sending graph data to /api/graph/view');

        let remoteError = null;
        const backendPayload = sanitizeGraphForBackend(rawJson);
        try {
            const response = await fetch('/api/graph/view', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(backendPayload)
            });

            let payload;
            try {
                payload = await response.json();
            } catch (parseError) {
                if (!response.ok) {
                    throw new Error(`Backend responded with ${response.status}`);
                }
                throw parseError;
            }

            if (!response.ok) {
                console.warn('[kg-gen] Graph preparation failed', payload);
                let message = payload && typeof payload === 'object'
                    ? payload.detail || payload.error || payload.message
                    : null;
                if (Array.isArray(payload?.detail)) {
                    message = payload.detail
                        .map(item => item.msg || item.message || JSON.stringify(item))
                        .join('; ');
                }
                throw new Error(message || 'Failed to prepare graph');
            }

            console.info('[kg-gen] Graph preparation succeeded');
            await renderView(payload.view, payload.graph);
            return;
        } catch (error) {
            remoteError = error;
            console.warn('[kg-gen] Falling back to local rendering', error);
            document.getElementById('errorGraphMessage').classList.add('show');
            hideFloatingActions();
            setStatus(`Failed to load graph: ${error.message}`, 'error');
            hideLoadingInViewer();
            resetViewer();
        }

        try {
            const viewModel = buildViewModelLocally(rawJson);
            console.info('[kg-gen] Graph rendered locally');
            await renderView(viewModel, rawJson);
            setStatus(`${extractSummary(viewModel)} (rendered locally)`, 'success');
        } catch (fallbackError) {
            console.error('[kg-gen] Local conversion failed', fallbackError);
            const combinedMessage = fallbackError?.message || remoteError?.message || 'Unknown error';
            setStatus(`Could not load graph: ${combinedMessage}`, 'error');
            document.getElementById('errorGraphMessage').classList.add('show');
            hideFloatingActions();
            hideLoadingInViewer();
            resetViewer();
            throw fallbackError;
        }
    }

    async function handleGraphFile(file) {
        showFloatingActions();

        if (!file) {
            return;
        }

        if (!confirmGraphReplacement('file upload')) {
            return;
        }

        setStatus(`Reading ${file.name}...`);
        console.info('[kg-gen] Reading uploaded graph file', file.name);

        // Hide mobile sidebar and show loading with proper timing
        if (window.sidebarManager && window.sidebarManager.isMobile) {
            window.sidebarManager.hideMobileSidebar();
            // Give sidebar animation time to complete before showing loading
            setTimeout(() => {
                showLoadingInViewer('Loading Graph', `Reading ${file.name}...`);
            }, 150);
        } else {
            showLoadingInViewer('Loading Graph', `Reading ${file.name}...`);
        }

        try {
            const contents = await readFile(file);
            const json = JSON.parse(contents);
            await handleGraphData(json);
        } catch (error) {
            console.error(error);
            document.getElementById('errorGraphMessage').classList.add('show');
            hideFloatingActions();
            setStatus(`Could not load graph: ${error.message}`, 'error');
            hideFloatingActions();
            hideLoadingInViewer();
            resetViewer();
        }
    }

    async function hideFloatingActions() {
        const floatingActions = document.getElementById('floatingActions');
        if (floatingActions) {
            floatingActions.style.display = 'none';
        }
    }

    async function showFloatingActions() {
        const floatingActions = document.getElementById('floatingActions');
        if (floatingActions) {
            floatingActions.style.display = 'flex';
        }
    }

    async function generateGraph() {

        hideGenerateError();

        const apiKey = apiKeyInput.value.trim();
        const pastedText = sourceText.value.trim();
        const textFile = textFileInput.files?.[0];
        const chunkSizeValue = chunkSizeInput.value.trim();
        const temperatureValue = temperatureInput.value.trim();

        if (!apiKey) {
            const errorMessage = 'Enter your OpenAI API key to generate a graph.';      
            setStatus(errorMessage, 'error');
            showGenerateError(errorMessage);
            return;
        }

        if (!pastedText && !textFile) {
            const errorMessage = 'Provide some text or upload a .txt file.';
            setStatus(errorMessage, 'error');
            showGenerateError(errorMessage);
            return;
        }

        if (!confirmGraphReplacement('graph generation')) {
            const errorMessage = 'Please confirm the graph generation.';
            setStatus(errorMessage, 'error');
            showGenerateError(errorMessage);
            return;
        }

        isGenerating = true;
        generateButton.disabled = true;
        generateButton.textContent = 'Generating...';

        const formData = new FormData();
        formData.append('api_key', apiKey);
        formData.append('model', modelSelect.value);
        formData.append('cluster', clusterToggle.checked ? 'true' : 'false');
        formData.append('retrieval_model', retrievalModelSelect.value);
        if (pastedText) {
            formData.append('source_text', pastedText);
        }
        if (textFile) {
            formData.append('text_file', textFile);
        }
        if (contextInput.value.trim()) {
            formData.append('context', contextInput.value.trim());
        }
        if (chunkSizeValue) {
            formData.append('chunk_size', chunkSizeValue);
        }
        if (temperatureValue) {
            formData.append('temperature', temperatureValue);
        }

        setStatus('Generating graph with KGGen...');
        console.info('[kg-gen] Submitting generate request', {
            model: modelSelect.value,
            cluster: clusterToggle.checked,
            retrievalModel: retrievalModelSelect.value,
            hasText: Boolean(pastedText),
            hasFile: Boolean(textFile)
        });

        showLoadingInViewer('Generating Graph', 'Running KGGen on your text. This may take a few minutes...');
        if (!hasLoadedGraph) {
            resetViewer();
        }

        try {
            const response = await fetch('/api/generate', {
                method: 'POST',
                body: formData
            });

            const rawBody = await response.text();
            let payload = null;
            if (rawBody) {
                try {
                    payload = JSON.parse(rawBody);
                } catch (parseError) {
                    console.warn('[kg-gen] Response was not valid JSON', parseError);
                }
            }

            if (!response.ok) {
                console.warn('[kg-gen] Generation endpoint returned error', payload || rawBody);
                const message =
                    (payload && (payload.detail || payload.error || payload.message)) ||
                    rawBody?.trim() ||
                    `Generation failed (${response.status})`;
                throw new Error(message);
            }

            if (!payload || typeof payload !== 'object') {
                throw new Error('Generation succeeded but returned invalid response.');
            }

            console.info('[kg-gen] Generation succeeded');
            hideGenerateError();
            await renderView(payload.view, payload.graph);
        } catch (error) {
            console.error(error);
            const errorMessage = 'Generation failed: ' + (error.message || 'Unknown error');
            setStatus(`Generation failed: ${error.message}`, 'error');
            showGenerateError(errorMessage);
            hideLoadingInViewer();
        } finally {
            isGenerating = false;
            generateButton.disabled = false;
            generateButton.textContent = 'Generate graph';
        }
    }


    async function onChangeSelectExamples(){
        if (exampleSelect) {
            exampleSelect.addEventListener('change', async event => {
                if (modalOpenGraphViewer) {
                modalOpenGraphViewer.close();
                }
                const slug = event.target.value;
                if (!slug) {
                    exampleStatus.textContent = exampleMetadata.size
                        ? 'Select an example to load it instantly.'
                        : exampleStatus.textContent;
                    updateExampleLink(null);
                    return;
                }
    
                if (!confirmGraphReplacement('example loading')) {
                    exampleSelect.value = '';
                    return;
                }
    
                const meta = exampleMetadata.get(slug) || null;
                updateExampleLink(meta);
                const title = meta?.title || slug;
                exampleStatus.textContent = `Loading ${title}...`;
                setStatus(`Loading example graph: ${title}...`);
    
                // Hide mobile sidebar and show loading with proper timing
                if (window.sidebarManager && window.sidebarManager.isMobile) {
                    window.sidebarManager.hideMobileSidebar();
                    // Give sidebar animation time to complete before showing loading
                    setTimeout(() => {
                        showLoadingInViewer('Loading Example', `Loading ${title}...`);
                    }, 150);
                } else {
                    showLoadingInViewer('Loading Example', `Loading ${title}...`);
                }
                if (!hasLoadedGraph) {
                    resetViewer();
                }
    
                exampleSelect.disabled = true;
                try {
                    const response = await fetch(`/api/examples/${slug}`);
                    let payload;
                    try {
                        payload = await response.json();
                    } catch (parseError) {
                        if (response.ok) {
                            throw new Error('Example payload is not valid JSON');
                        }
                        throw new Error(`Request failed (${response.status})`);
                    }
    
                    if (!response.ok) {
                        const message = payload?.detail || payload?.error || `Failed to load example (${response.status})`;
                        throw new Error(message);
                    }
    
                    await handleGraphData(payload);
                    exampleStatus.textContent = `Loaded ${title}.`;
                } catch (error) {
                    console.error('[kg-gen] Failed to load example graph', error);
                    setStatus(`Failed to load example '${title}': ${error.message}`, 'error');
                    exampleStatus.textContent = 'Could not load the selected sample.';
                    hideLoadingInViewer();
                } finally {
                    exampleSelect.disabled = exampleMetadata.size === 0;
                }
            });
        }
    }

    if (modelSelect) {
        applyDefaultTemperature(modelSelect.value, false);
        modelSelect.addEventListener('change', event => {
            applyDefaultTemperature(event.target.value, true);
        });
    }

   async function onChangeDropZone() {
       ['dragenter', 'dragover'].forEach(eventName => {
        graphDropZone.addEventListener(eventName, event => {
            event.preventDefault();
            event.stopPropagation();
            graphDropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        graphDropZone.addEventListener(eventName, event => {
            event.preventDefault();
            event.stopPropagation();
            graphDropZone.classList.remove('dragover');
        });
    });

    graphDropZone.addEventListener('drop', event => {
        const files = event.dataTransfer?.files;
        if (files && files.length) {
            console.info('[kg-gen] File dropped', files[0].name);
            handleGraphFile(files[0]);
        }
    });
   }

    async function onChangeGenerateButton() {
        if (generateButton) {
        generateButton.addEventListener('click', event => {
            event.preventDefault();
            generateGraph();
        });
        }
    }

    async function onChangeClearTextButton() {
        clearTextButton.addEventListener('click', event => {
            event.preventDefault();
            sourceText.value = '';
            textFileInput.value = '';
            // Reset drop zone text
            updateDropZoneText(null);
            // Reset textarea state
            toggleTextareaState(false);
            toggleDropZoneState(false);
            // Hide error message
            hideGenerateError();
            // Update states after clearing
            if (window.updateInputStates) {
                window.updateInputStates();
            }
            // Note: Cached inputs (api key, model, chunk size, temperature, cluster, context) are preserved
            setStatus('Text inputs cleared. Cached settings preserved.');
        });
    }

    downloadButton.addEventListener('click', () => {
        if (!lastGraphPayload) {
            return;
        }
        const blob = new Blob([JSON.stringify(lastGraphPayload, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'graph.json';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    });

    refreshButton.addEventListener('click', () => {
        if (!refreshCallbacks.length) {
            return;
        }
        const latest = refreshCallbacks[refreshCallbacks.length - 1];
        latest?.();
    });

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            console.log('[kg-gen] DOM loaded, initializing caching...');
            loadCachedInputs();
            setupInputCaching();
        });
    } else {
        // DOM is already loaded
        console.log('[kg-gen] DOM already loaded, initializing caching...');
        loadCachedInputs();
        setupInputCaching();
    }

    loadTemplate().catch(() => {
        /* Status already updated in loadTemplate */
    });
    loadInitialExampleGraph().catch(() => {
        /* Errors handled inside loadExamples */
    });

    // Global functions for action buttons
    window.openGraphViewer = function() {
        // For now, we'll show a simple modal
        modalOpenGraphViewer = new Modal('graphViewerModal', 'Open existing graph', null, {
            selector: '#selectorModalTemplate',
            width: '800px',
            closable: true,
            backdrop: true
        });

        graphFileInput = document.getElementById('graphFile');
        graphDropZone = document.getElementById('graphDropZone');
        exampleSelect = document.getElementById('exampleGraph');
        exampleLink = document.getElementById('exampleLink');
        exampleStatus = document.getElementById('exampleStatus');

        graphDropZone.addEventListener('click', event => {
            event.preventDefault();
            graphFileInput.click();
        });

        graphFileInput.addEventListener('change', event => {
            const [file] = event.target.files;
            handleGraphFile(file);
            graphFileInput.value = '';
            modalOpenGraphViewer.close();
        });

        graphDropZone.addEventListener('keydown', event => {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                graphFileInput.click();
            }
        });

        onChangeSelectExamples()
        onChangeDropZone()
        window.loadExamples();
    };

     // Error message functions
     function showGenerateError(message) {
        const errorMessage = document.getElementById('generateErrorMessage');
        const errorText = document.getElementById('generateErrorText');
        
        if (errorMessage && errorText) {
            errorText.textContent = message;
            errorMessage.style.display = 'flex';
            errorMessage.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }

    function hideGenerateError() {
        const errorMessage = document.getElementById('generateErrorMessage');
        if (errorMessage) {
            errorMessage.style.display = 'none';
        }
    }

     function updateDropZoneText(filename) {
        const dropZoneText = document.getElementById('dropZoneText');
        const dropZoneHint = document.getElementById('dropZoneHint');
        const fileRemoveBtn = document.getElementById('fileRemoveBtn');
        
        if (dropZoneText && dropZoneHint && fileRemoveBtn) {
            if (filename) {
                dropZoneText.textContent = `Loaded ${filename}`;
                dropZoneHint.textContent = 'Click to choose a different file';
                fileRemoveBtn.style.display = 'flex';
            } else {
                dropZoneText.textContent = 'Drop file here or click to choose';
                dropZoneHint.textContent = 'Accepts .txt files';
                fileRemoveBtn.style.display = 'none';
            }
        }
    }

        // Function to toggle drop zone disabled state
        function toggleDropZoneState(hasText) {
        const dropZone = document.getElementById('graphDropZoneText');
        const dropZoneDisabledMessage = document.getElementById('dropZoneDisabledMessage');
        const textFileInput = document.getElementById('textFile');
        
        if (dropZone && dropZoneDisabledMessage) {
            if (hasText) {
                // Disable drop zone
                dropZone.classList.add('disabled');
                dropZone.setAttribute('tabindex', '-1');
                dropZone.setAttribute('aria-disabled', 'true');
                dropZoneDisabledMessage.style.display = 'block';
                
                // Clear any existing file
                if (textFileInput) {
                    textFileInput.value = '';
                    updateDropZoneText(null);
                }
            } else {
                // Enable drop zone
                dropZone.classList.remove('disabled');
                dropZone.setAttribute('tabindex', '0');
                dropZone.setAttribute('aria-disabled', 'false');
                dropZoneDisabledMessage.style.display = 'none';
            }
        }
    }

        // Function to toggle textarea disabled state
    function toggleTextareaState(hasFile) {
        const textarea = document.getElementById('sourceText');
        const textareaDisabledMessage = document.getElementById('sourceTextDisabledMessage');
        
        if (textarea && textareaDisabledMessage) {
            if (hasFile) {
                // Disable textarea
                textarea.disabled = true;
                textarea.setAttribute('aria-disabled', 'true');
                textareaDisabledMessage.style.display = 'block';
                
                // Clear any existing text
                textarea.value = '';
            } else {
                // Enable textarea
                textarea.disabled = false;
                textarea.removeAttribute('aria-disabled');
                textareaDisabledMessage.style.display = 'none';
            }
        }
    }

    window.generateFromText = function() {
        // Add your text generation logic here
        // For now, we'll show a simple modal
        modalTextGenerator = new Modal('textGeneratorModal', 'Generate from text', null, {
            selector: '#textGeneratorModalTemplate',
            width: '800px',
            closable: true,
            backdrop: true
        });


        apiKeyInput = document.getElementById('apiKey');
        modelSelect = document.getElementById('model');
        chunkSizeInput = document.getElementById('chunkSize');
        temperatureInput = document.getElementById('temperature');
        clusterToggle = document.getElementById('clusterToggle');
        retrievalModelSelect = document.getElementById('retrievalModel');
        contextInput = document.getElementById('context');
        sourceText = document.getElementById('sourceText');
        textFileInput = document.getElementById('textFile');
        generateButton = document.getElementById('generateButton');
        clearTextButton = document.getElementById('clearTextButton');
        

        onChangeGenerateButton()
        onChangeClearTextButton()

       

        // Function to remove selected file
        function removeSelectedFile() {
            const textFileInput = document.getElementById('textFile');
            if (textFileInput) {
                textFileInput.value = '';
                updateDropZoneText(null);
                toggleTextareaState(false);
            }
        }

        // Initialize text file drop zone
        const graphDropZoneText = document.getElementById('graphDropZoneText');
        const fileRemoveBtn = document.getElementById('fileRemoveBtn');
        
        // Add event listener for remove button
        if (fileRemoveBtn) {
            fileRemoveBtn.addEventListener('click', function(event) {
                event.preventDefault();
                event.stopPropagation(); // Prevent triggering drop zone click
                removeSelectedFile();
            });
        }
        
        if (graphDropZoneText) {
            // Click event for text file drop zone
            graphDropZoneText.addEventListener('click', event => {
                event.preventDefault();
                if (!graphDropZoneText.classList.contains('disabled')) {
                    textFileInput.click();
                }
            });

            // Keyboard event for text file drop zone
            graphDropZoneText.addEventListener('keydown', event => {
                if ((event.key === 'Enter' || event.key === ' ') && !graphDropZoneText.classList.contains('disabled')) {
                    event.preventDefault();
                    textFileInput.click();
                }
            });

            // Drag and drop events for text file
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                graphDropZoneText.addEventListener(eventName, event => {
                    if (!graphDropZoneText.classList.contains('disabled')) {
                        event.preventDefault();
                        event.stopPropagation();
                    }
                });
            });

            ['dragenter', 'dragover'].forEach(eventName => {
                graphDropZoneText.addEventListener(eventName, event => {
                    if (!graphDropZoneText.classList.contains('disabled')) {
                        graphDropZoneText.classList.add('dragover');
                    }
                });
            });

            ['dragleave', 'drop'].forEach(eventName => {
                graphDropZoneText.addEventListener(eventName, event => {
                    graphDropZoneText.classList.remove('dragover');
                });
            });

            graphDropZoneText.addEventListener('drop', event => {
                if (!graphDropZoneText.classList.contains('disabled')) {
                    const [file] = event.dataTransfer.files;
                    if (file && file.type === 'text/plain') {
                        // Create a FileList-like object and assign it to the input
                        const dataTransfer = new DataTransfer();
                        dataTransfer.items.add(file);
                        textFileInput.files = dataTransfer.files;
                        
                        // Update drop zone text to show loaded filename
                        updateDropZoneText(file.name);
                        
                        // Disable textarea
                        toggleTextareaState(true);
                        
                        // Trigger change event to update the UI
                        textFileInput.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                }
            });
        }

        // Add change event listener for text file input
        if (textFileInput) {
            textFileInput.addEventListener('change', event => {
                const file = event.target.files[0];
                if (file) {
                    updateDropZoneText(file.name);
                    toggleTextareaState(true);
                } else {
                    updateDropZoneText(null);
                    toggleTextareaState(false);
                }
            });
        }

        // Add event listeners for textarea to control drop zone state
        if (sourceText) {
            sourceText.addEventListener('input', function() {
                const hasText = this.value.trim().length > 0;
                toggleDropZoneState(hasText);
            });
            
            // Check initial state
            const hasText = sourceText.value.trim().length > 0;
            toggleDropZoneState(hasText);
        }

       

        // Password toggle functionality
        const passwordToggle = document.getElementById('passwordToggle');
        
        if (passwordToggle && apiKeyInput) {
            passwordToggle.addEventListener('click', function() {
                const isPassword = apiKeyInput.type === 'password';
                apiKeyInput.type = isPassword ? 'text' : 'password';
                
                // Update the eye icon
                const eyeIcon = passwordToggle.querySelector('.eye-icon');
                if (eyeIcon) {
                    if (isPassword) {
                        // Show eye with slash (hidden)
                        eyeIcon.innerHTML = '<path d="M12 7c2.76 0 5 2.24 5 5 0 .65-.13 1.26-.36 1.83l2.92 2.92c1.51-1.26 2.7-2.89 3.43-4.75-1.73-4.39-6-7.5-11-7.5-1.4 0-2.74.25-3.98.7l2.16 2.16C10.74 7.13 11.35 7 12 7zM2 4.27l2.28 2.28.46.46C3.08 8.3 1.78 10.02 1 12c1.73 4.39 6 7.5 11 7.5 1.55 0 3.03-.3 4.38-.84l.42.42L19.73 22 21 20.73 3.27 3 2 4.27zM7.53 9.8l1.55 1.55c-.05.21-.08.43-.08.65 0 1.66 1.34 3 3 3 .22 0 .44-.03.65-.08l1.55 1.55c-.67.33-1.41.53-2.2.53-2.76 0-5-2.24-5-5 0-.79.2-1.53.53-2.2zm4.31-.78l3.15 3.15.02-.16c0-1.66-1.34-3-3-3l-.17.01z" fill="currentColor"/>';
                        passwordToggle.setAttribute('aria-label', 'Hide password');
                    } else {
                        // Show regular eye (visible)
                        eyeIcon.innerHTML = '<path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5zm0-8c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3z" fill="currentColor"/>';
                        passwordToggle.setAttribute('aria-label', 'Show password');
                    }
                }
            });
        }


    };

    // Global search shortcut functionality
    function initializeSearchShortcut() {
        function handleSearchShortcut(event) {
            // Check for Cmd+F (Mac) or Ctrl+F (Windows/Linux)
            if ((event.metaKey || event.ctrlKey) && event.key === 'f') {
                event.preventDefault(); // Prevent browser's default find dialog
                
                // Try to find search inputs in parent window first
                let searchInput = null;
                let mobileSearchInput = null;
                
                try {
                    // Access parent window's document (for iframe scenarios)
                    if (window.parent && window.parent.document) {
                        searchInput = window.parent.document.getElementById('globalSearch');
                        mobileSearchInput = window.parent.document.getElementById('globalSearch-mobile');
                    }
                } catch (e) {
                    // Cross-origin iframe, can't access parent
                }
                
                // If not found in parent, try current document
                if (!searchInput) {
                    searchInput = document.getElementById('globalSearch');
                }
                if (!mobileSearchInput) {
                    mobileSearchInput = document.getElementById('globalSearch-mobile');
                }
                
                // Try desktop search first, then mobile
                if (searchInput && !searchInput.disabled) {
                    searchInput.focus();
                    searchInput.select(); // Select all text for easy replacement
                } else if (mobileSearchInput && !mobileSearchInput.disabled) {
                    mobileSearchInput.focus();
                    mobileSearchInput.select();
                }
            }
        }
        
        // Add event listener to current document
        document.addEventListener('keydown', handleSearchShortcut);
        
        // Also add to parent document if we're in an iframe
        try {
            if (window.parent && window.parent !== window && window.parent.document) {
                window.parent.document.addEventListener('keydown', handleSearchShortcut);
            }
        } catch (e) {
            // Cross-origin iframe, can't access parent document
        }
    }

    // Initialize search shortcut and iframe handling when DOM is loaded
    document.addEventListener('DOMContentLoaded', function() {
        initializeSearchShortcut();
        setupIframeKeyboardHandling();
    });

    // Also initialize immediately in case DOM is already loaded
    if (document.readyState === 'loading') {
        // DOM is still loading, the DOMContentLoaded event will handle it
    } else {
        // DOM is already loaded
        initializeSearchShortcut();
        setupIframeKeyboardHandling();
    }

    // Enhanced iframe focus management and keyboard event handling
    function setupIframeKeyboardHandling() {
        const iframe = document.getElementById('viewer');
        
        if (!iframe) {
            console.warn('[kg-gen] iframe with id "viewer" not found');
            return;
        }

        // Ensure the iframe can be focused
        iframe.removeAttribute('hidden');
        iframe.setAttribute('tabindex', '-1');
        
        iframe.addEventListener('load', () => {
            try {
                // Access iframe's document (requires same-origin)
                const doc = iframe.contentDocument || iframe.contentWindow.document;
                
                if (doc) {
                    // Add keyboard event listener to iframe's document
                    doc.addEventListener('keydown', (e) => {
                        // Handle Ctrl+F / Cmd+F for search
                        const isCtrlF = (e.ctrlKey || e.metaKey) && (e.key.toLowerCase() === 'f');
                        if (isCtrlF) {
                            e.preventDefault();
                            // Try to focus search inputs in parent window
                            const searchInput = document.getElementById('globalSearch');
                            const mobileSearchInput = document.getElementById('globalSearch-mobile');
                            
                            if (searchInput && !searchInput.disabled) {
                                searchInput.focus();
                                searchInput.select();
                            } else if (mobileSearchInput && !mobileSearchInput.disabled) {
                                mobileSearchInput.focus();
                                mobileSearchInput.select();
                            }
                            return false;
                        }
                        
                        // Handle Escape key to close any open modals
                        if (e.key === 'Escape') {
                            // Let the event bubble up to parent window
                            // The modal system will handle closing modals
                        }
                        
                        // Handle Tab key for focus management within iframe
                        if (e.key === 'Tab') {
                            // Let the iframe handle its own tab navigation
                        }
                    }, { capture: true });
                    
                    // Focus the iframe when it loads
                    iframe.focus();
                    
                    console.log('[kg-gen] iframe keyboard handling setup complete');
                } else {
                    console.warn('[kg-gen] Cannot access iframe content - may be cross-origin');
                }
            } catch (error) {
                console.warn('[kg-gen] Error setting up iframe keyboard handling:', error);
                console.warn('[kg-gen] This is normal if the iframe content is cross-origin');
            }
        });
        
        // Also handle focus when iframe becomes visible
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.type === 'attributes' && mutation.attributeName === 'hidden') {
                    const iframe = mutation.target;
                    if (!iframe.hasAttribute('hidden')) {
                        console.log('[kg-gen] iframe became visible, focusing');
                        setTimeout(() => {
                            iframe.focus();
                        }, 100);
                    }
                }
            });
        });
        
        observer.observe(iframe, { attributes: true });
        
        console.log('[kg-gen] iframe focus management setup complete');
    }

    window.loadExamples = loadExamples;
    window.loadInitialExampleGraph = loadInitialExampleGraph;
    window.hideGenerateError = hideGenerateError;
    window.showGenerateError = showGenerateError;
    window.updateDropZoneText = updateDropZoneText;
    window.toggleDropZoneState = toggleDropZoneState;
    window.toggleTextareaState = toggleTextareaState;
})();

