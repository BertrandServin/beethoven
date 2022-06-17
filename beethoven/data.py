"""Handles data input / output 

This modules handles specific file formats used in beethoven.

"""
from collections import OrderedDict
import numpy as np
import pandas as pd


def snplist_from_file(filename):
    """
    Reads a list of SNP from file with columns 'CHROM','POS' and optionally 'SNP'.

    Columns must be whitespace separated.

    If column 'SNP' exists returns it
    else returns SNP IDs created by CHROM.POS
    """
    dat = pd.read_csv(str(filename), delim_whitespace=True)
    ## SNPs
    try:
        snpnames = dat['SNP']
    except KeyError:
        snpnames = [':'.join( vals) for vals in zip( dat['CHROM'].astype(str), dat['POS'].astype( str))]
    return set(snpnames)

class WorkerPoolseqData:
    '''Data from a poolseq experiment of workers in a hive.

    The class should be instanciated using one of the classmethods:

    - WorkerPoolseqData.from_count_depth

    '''
    
    def __init__( self):
        self._x = None
        self._d = None
        self._snps = None
        self.origin="an unknown source"

    def __str__( self):
        return "Worker Poolseq Data from {}, contains data on {} markers".format( self.origin, len( self.snps))
    
    @property
    def x( self):
        '''
        :obj:`numpy.array` of :obj:`int`: 1D vector of reference allele counts
        '''
        return self._x
    @x.setter
    def x( self, value):
        if self._x is None:
            self._x = np.array( value)
        else:
            print( 'Not modifying original data, period.')

    @property
    def d( self):
        '''
        :obj:`numpy.array` of :obj:`int`: 1D Vector of sequencing depth
        '''
        return self._d
    @d.setter
    def d( self, value):
        if self._d is None:
            self._d = np.array( value)
        else:
            print('Not modifying original data, period.')

    @property
    def snps( self):
        '''
        :obj:`collections.OrderedDict`: keys are names (chr.pos / rsnumber ...), values (chr, pos, index) tuples
        '''
        return self._snps
    @snps.setter
    def snps( self, value):
        if self._snps is None:
            try:
                assert isinstance( value, OrderedDict)
            except AssertionError:
                raise TypeError( 'SNPs must be provided as a collections.OrderedDict instance')
            self._snps = value
        else:
            print( 'Not modifying original data, period.')
    @property
    def nsnps( self):
        """Number of SNPs in the data"""
        return len( self._snps)
    @property
    def snplist( self):
        """List of SNP identifiers (order is conserved)"""
        return self.snps.keys()

    def get_data( self, snplist):
        """Extract data.

        Parameters
        ----------
        snplist : list of str
                  List of SNP identifiers to extract

        Returns
        -------
        Dictionary with keys:
        x : numpy.array of int
            Reference allele counts 
        d : numpy.array of int
            Sequencing dephts

        Notes
        -----
        The return values respect the order in `snplist`
        
        If one identifier is not found, the respective values in `x` and `d` are 0 and 0.
        """
        nquery = len( snplist)
        x_ret = np.zeros( nquery, dtype=np.int)
        d_ret = np.zeros( nquery, dtype=np.int)
        for i,s in enumerate( snplist):
            try:
                idx = self.snps[s][2]
                x_ret[i] = self.x[ idx]
                d_ret[i] = self.d[ idx]
            except KeyError:
                continue
            except ValueError:
                print(i,s,self.snps[s])
                raise
        return { 'x':x_ret, 'd':d_ret}

    @classmethod
    def from_count_depth( cls, filename, **kwargs):
        '''Instantiate using data from filename in depth/count format.
        
        Parameters
        ----------
        filename : str
            Name of the file to read.
        **kwargs : 
            keyword arguments passed to pandas.read_csv
        

        Format specification (column names matter, delim is space):
        
        .. code-block:: text

            CHROM POS D X
            CM009931.1 3242992 4 4
            CM009931.1 4446365 24 19
            CM009931.1 10644065 22 3
            CM009931.1 11107973 24 11
            CM009931.1 11606557 26 18
            ...

        The input file may include an additional column 'SNP' with SNP
        identifiers. If not the SNP names will be set to
        CHR.str(POS). Data is read using pandas.read_csv.

        '''
        obj = cls()
        obj.origin = str(filename)
        dat = pd.read_csv(str(filename), delim_whitespace=True)
        dat.dropna(inplace=True)
        ## SNPs
        try:
            snpnames = dat['SNP']
        except KeyError:
            snpnames = [':'.join( vals) for vals in zip( dat['CHROM'].astype(str), dat['POS'].astype( str))]
        snpvalues = zip(dat['CHROM'], dat['POS'], range( dat.shape[0]))
        obj.snps = OrderedDict(zip(snpnames, snpvalues))
        obj.x = np.array(dat['X'])
        obj.d = np.array(dat['D'])
        return obj
    

class PanelFreqData:
    """Data on allele frequencies in different groups from a diversity panel.

    Attributes
    ----------

    group_names : tuple of str
        Names of groups 
    snps : OrderedDict of SNPs: 
        keys are names values are (chr, pos, idx)
    F : np.array of float
        matrix of allele frequencies
    """

    def __init__(self):
        self._gnames = None
        self._snps = None
        self._F = None
        self.origin = "an unknown source"
        
    def __str__(self):
        return "Frequency data from {}, contains data on {} markers in {} groups".format(self.origin, self.nsnps, self.ngroups)

    @property
    def group_names( self):
        """
        Tuple of group names
        """
        return self._gnames
    @group_names.setter
    def group_names(self, value):
        if self._gnames is None:
            self._gnames = tuple(value)
        else:
            print('Not modifying original data, period.')
    @property
    def ngroups(self):
        return len(self._gnames)

    @property
    def snps(self):
        '''
        OrderedDict of SNPs: keys are names (chr.pos / rsnumber ...), values (chr, pos, index) tuples
        '''
        return self._snps
    @snps.setter
    def snps( self, value):
        if self._snps is None:
            try:
                assert isinstance( value, OrderedDict)
            except AssertionError:
                raise TypeError('SNPs must be provided as a collections.OrderedDict instance')
            self._snps = value
        else:
            print('Not modifying original data, period.')
    @property
    def nsnps(self):
        return len(self._snps)
    @property
    def snplist(self):
        return self.snps.keys()


    @property
    def F(self):
        return self._F
    @F.setter
    def F(self, value):
        if self._F is None:
            val = np.array(value, dtype=np.float)
            assert len(val.shape) == 2
            self._F = val
        else:
            print( 'Not modifying original data, period.')
        
    @classmethod
    def from_file( cls, filename, minvar = 0.01):
        ''' Instantiate using data from filename in matrix format.

        Format specification (column names matter, delim is space):

        .. code-block:: text

            CHROM POS Ligustica Mellifera Caucasica
            CM009940.1 7987618 0.977777777777778 0.956043956043956 0
            CM009931.1 11606557 0.976190476190476 0.0194174757281553 1
            CM009941.1 2165159 0.0112359550561798 0.494736842105263 0
            CM009941.1 1050223 0.0659340659340659 0.805825242718447 1
            CM009937.1 1177419 0.263736263736264 1 0
            CM009931.1 17216362 0.934782608695652 1 0
            ...

        The input file may include an additional column 'SNP' with SNP identifiers. If not 
        the SNP namesclassmet will be set to CHR.str(POS). Data is read using pandas.read_csv.
        '''
        obj = cls()
        obj.origin = str( filename)
        dat = pd.read_csv( str( filename), delim_whitespace = True)
        dat.dropna(inplace=True)
        ## groups
        subcols = [ not (x in ['CHROM','POS','SNP']) for x in dat.columns]
        obj.group_names = dat.columns[ subcols]
        ## F matrix
        fmat_tmp = np.array(dat.iloc[ :, subcols])
        vf = fmat_tmp.var(axis = 1)
        snpsel = np.isfinite(vf) 
        dat = dat.loc[ snpsel,:]
        fmat = np.array( dat.iloc[ :, subcols])
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore', r'invalid value encountered in greater')
            np.warnings.filterwarnings('ignore', r'invalid value encountered in less')
            fmat = np.where( fmat > 0.999, 0.99, fmat)
            fmat = np.where( fmat < 0.001, 0.01, fmat)
        obj.F = fmat
        ## SNPs
        try:
            snpnames = dat['SNP']
        except KeyError:
            snpnames = [ ':'.join( vals) for vals in zip( dat['CHROM'].astype(str), dat['POS'].astype( str))]
        snpvalues = zip( dat['CHROM'], dat['POS'], range( dat.shape[0]))
        obj.snps = OrderedDict( zip( snpnames, snpvalues))
        return obj

    def get_data( self, snplist):
        """ Get data (x,d) at SNPs listed in snplist.

        If SNP is not found raises ValueError
        """
        snpidx = []
        nquery = len(snplist)
        ret_f = np.full_like(np.empty( (nquery, self.F.shape[1]), dtype=np.float64), fill_value=np.nan)
        for i,s in enumerate( snplist):
            try:
                ret_f[i,] = self.F[self.snps[s][2],]
            except KeyError:
                continue
        return { 'F' : ret_f }
